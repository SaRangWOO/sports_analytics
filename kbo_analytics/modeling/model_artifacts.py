from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn


ARTIFACT_VERSION = "1"
SCHEMA_VERSION = "1"
ARTIFACT_FILES = ("model.joblib", "metadata.json", "feature_schema.json", "metrics.json")
ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ArtifactValidationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"JSON 파일을 읽을 수 없습니다: {path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 최상위 값은 객체여야 합니다: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ArtifactValidationError("아티팩트 루트 밖의 경로는 사용할 수 없습니다.")
    return path


def _artifact_id(value: str) -> str:
    if not ARTIFACT_ID_PATTERN.fullmatch(value):
        raise ArtifactValidationError("유효하지 않은 artifact_id입니다.")
    return value


def _version_prefix(value: str, parts: int) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    if len(numbers) < parts:
        raise ArtifactValidationError(f"버전 값을 해석할 수 없습니다: {value}")
    return tuple(int(number) for number in numbers[:parts])


def _runtime_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
    }


def _code_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _previous_production_id(root: Path) -> str | None:
    metadata_path = root / "production" / "current" / "metadata.json"
    if not metadata_path.exists():
        return None
    return _read_json(metadata_path).get("artifact_id")


def _manifest_payload(directory: Path, artifact_id: str, approval_status: str, previous_artifact_id: str | None) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": ARTIFACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "files": [{"name": name, "sha256": _sha256(directory / name)} for name in ARTIFACT_FILES],
        "complete": True,
        "approval_status": approval_status,
        "created_at": utc_now(),
        "previous_artifact_id": previous_artifact_id,
    }


def _refresh_manifest(directory: Path, approval_status: str, previous_artifact_id: str | None) -> None:
    metadata = _read_json(directory / "metadata.json")
    metadata["approval_status"] = approval_status
    metadata["previous_artifact_id"] = previous_artifact_id
    _write_json(directory / "metadata.json", metadata)
    _write_json(
        directory / "manifest.json",
        _manifest_payload(directory, metadata["artifact_id"], approval_status, previous_artifact_id),
    )


def _class_order(bundle: dict[str, Any]) -> list[Any]:
    if bundle["model_type"] == "from_scratch_logistic_regression":
        return [0, 1]
    model = bundle.get("model")
    if model is None or not hasattr(model, "classes_"):
        raise ArtifactValidationError("학습 모델의 class 순서를 확인할 수 없습니다.")
    return [value.item() if hasattr(value, "item") else value for value in model.classes_]


def create_candidate_artifact(
    artifact_root: str | Path,
    prediction_bundle: dict[str, Any],
    metadata: dict[str, Any],
    feature_schema: dict[str, Any],
    metrics: dict[str, Any],
    project_root: str | Path,
) -> Path:
    root = Path(artifact_root).resolve()
    candidate_root = root / "candidate"
    candidate_root.mkdir(parents=True, exist_ok=True)
    bundle = dict(prediction_bundle)
    bundle["feature_order"] = list(feature_schema["feature_order"])
    bundle["prediction_unit"] = metadata["prediction_unit"]
    bundle["class_order"] = _class_order(bundle)
    fingerprint = joblib.hash(
        {
            "bundle": bundle,
            "model_name": metadata["model_name"],
            "training_cutoff_date": metadata["training_cutoff_date"],
            "feature_order": bundle["feature_order"],
        }
    )
    artifact_id = _artifact_id(metadata.get("artifact_id") or f"{metadata['training_cutoff_date']}-{fingerprint[:16]}")
    target = candidate_root / artifact_id
    if target.exists():
        validate_artifact(root, target, expected_approval="candidate")
        return target

    temp_dir = Path(tempfile.mkdtemp(prefix=".candidate-", dir=candidate_root))
    try:
        joblib.dump(bundle, temp_dir / "model.joblib", compress=3)
        full_metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "artifact_id": artifact_id,
            "approval_status": "candidate",
            "created_at": utc_now(),
            "code_commit": _code_commit(Path(project_root)),
            "previous_artifact_id": _previous_production_id(root),
            **_runtime_versions(),
            **metadata,
        }
        full_metadata["artifact_id"] = artifact_id
        full_metadata["artifact_version"] = ARTIFACT_VERSION
        full_metadata["approval_status"] = "candidate"
        schema = {
            "schema_version": SCHEMA_VERSION,
            **feature_schema,
        }
        schema["schema_version"] = SCHEMA_VERSION
        _write_json(temp_dir / "metadata.json", full_metadata)
        _write_json(temp_dir / "feature_schema.json", schema)
        _write_json(temp_dir / "metrics.json", metrics)
        _write_json(
            temp_dir / "manifest.json",
            _manifest_payload(temp_dir, artifact_id, "candidate", full_metadata.get("previous_artifact_id")),
        )
        validate_artifact(root, temp_dir, expected_approval="candidate")
        os.replace(temp_dir, target)
        return target
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _validate_compatibility(metadata: dict[str, Any]) -> None:
    if _version_prefix(metadata["python_version"], 2) != _version_prefix(platform.python_version(), 2):
        raise ArtifactValidationError("Python major/minor 버전이 아티팩트와 호환되지 않습니다.")
    if _version_prefix(metadata["scikit_learn_version"], 2) != _version_prefix(sklearn.__version__, 2):
        raise ArtifactValidationError("scikit-learn major/minor 버전이 아티팩트와 호환되지 않습니다.")
    if _version_prefix(metadata["pandas_version"], 1) != _version_prefix(pd.__version__, 1):
        raise ArtifactValidationError("pandas major 버전이 아티팩트와 호환되지 않습니다.")
    if _version_prefix(metadata["numpy_version"], 1) != _version_prefix(np.__version__, 1):
        raise ArtifactValidationError("numpy major 버전이 아티팩트와 호환되지 않습니다.")


def validate_available_features(columns: list[str], schema: dict[str, Any]) -> None:
    required = list(schema["required_features"])
    optional = set(schema.get("optional_features", []))
    missing = [feature for feature in required if feature not in columns]
    unexpected = [feature for feature in columns if feature not in set(required) | optional]
    if missing:
        raise ArtifactValidationError(f"필수 피처가 누락되었습니다: {', '.join(missing)}")
    if unexpected:
        raise ArtifactValidationError(f"예상하지 못한 피처가 있습니다: {', '.join(unexpected)}")


def validate_model_input(frame: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    expected = list(schema["feature_order"])
    if list(frame.columns) != expected:
        raise ArtifactValidationError("모델 입력 피처 순서 또는 개수가 schema와 다릅니다.")
    converted = frame.apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        raise ArtifactValidationError("모델 입력에 숫자로 변환할 수 없는 값 또는 결측값이 있습니다.")
    return converted.astype(float)


def predict_bundle_probabilities(bundle: dict[str, Any], frame: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    model_input = validate_model_input(frame, schema)
    if bundle["model_type"] == "from_scratch_logistic_regression":
        values = model_input.to_numpy() @ np.asarray(bundle["weights"]) + float(bundle["bias"])
        values = np.clip(values, -30, 30)
        return 1 / (1 + np.exp(-values))
    model = bundle["model"]
    probabilities = model.predict_proba(model_input)
    classes = list(bundle["class_order"])
    if 1 not in classes:
        raise ArtifactValidationError("승리 class 1을 모델 class 순서에서 찾을 수 없습니다.")
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def validate_artifact(
    artifact_root: str | Path,
    artifact_dir: str | Path,
    expected_approval: str | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    directory = _trusted_path(root, Path(artifact_dir))
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactValidationError("manifest.json이 없습니다.")
    manifest = _read_json(manifest_path)
    if manifest.get("complete") is not True:
        raise ArtifactValidationError("완성되지 않은 아티팩트입니다.")
    if manifest.get("artifact_version") != ARTIFACT_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("지원하지 않는 artifact/schema 버전입니다.")
    if expected_approval and manifest.get("approval_status") != expected_approval:
        raise ArtifactValidationError(f"{expected_approval} 승인 상태의 아티팩트가 아닙니다.")
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or len(files) != len(ARTIFACT_FILES)
        or {item.get("name") for item in files if isinstance(item, dict)} != set(ARTIFACT_FILES)
    ):
        raise ArtifactValidationError("manifest 구성 파일 목록이 올바르지 않습니다.")
    for item in files:
        name = item["name"]
        if Path(name).name != name:
            raise ArtifactValidationError("manifest에 안전하지 않은 파일 경로가 있습니다.")
        path = directory / name
        if not path.is_file():
            raise ArtifactValidationError(f"아티팩트 구성 파일이 없습니다: {name}")
        if _sha256(path) != item.get("sha256"):
            raise ArtifactValidationError(f"체크섬이 일치하지 않습니다: {name}")

    metadata = _read_json(directory / "metadata.json")
    schema = _read_json(directory / "feature_schema.json")
    metrics = _read_json(directory / "metrics.json")
    required_metadata = {
        "artifact_id",
        "model_name",
        "model_family",
        "prediction_unit",
        "training_cutoff_date",
        "target_name",
        "feature_count",
        "python_version",
        "scikit_learn_version",
        "pandas_version",
        "numpy_version",
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        raise ArtifactValidationError(f"metadata 필드가 누락되었습니다: {', '.join(missing_metadata)}")
    if metadata.get("artifact_id") != manifest.get("artifact_id"):
        raise ArtifactValidationError("metadata와 manifest의 artifact_id가 다릅니다.")
    if metadata.get("approval_status") != manifest.get("approval_status"):
        raise ArtifactValidationError("metadata와 manifest의 승인 상태가 다릅니다.")
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("지원하지 않는 feature schema 버전입니다.")
    feature_order = schema.get("feature_order")
    if not isinstance(feature_order, list) or feature_order != schema.get("feature_names"):
        raise ArtifactValidationError("feature_names와 feature_order가 일치하지 않습니다.")
    required_features = schema.get("required_features")
    optional_features = schema.get("optional_features", [])
    if len(feature_order) != schema.get("feature_count"):
        raise ArtifactValidationError("피처 개수가 일치하지 않습니다.")
    if not isinstance(required_features, list) or not isinstance(optional_features, list):
        raise ArtifactValidationError("필수 또는 선택 피처 목록이 올바르지 않습니다.")
    if set(required_features) & set(optional_features):
        raise ArtifactValidationError("필수 피처와 선택 피처가 중복됩니다.")
    if not set(feature_order).issubset(set(required_features) | set(optional_features)):
        raise ArtifactValidationError("모델 피처가 필수 또는 선택 피처로 분류되지 않았습니다.")
    if metadata.get("feature_count") != len(feature_order):
        raise ArtifactValidationError("metadata와 feature schema의 피처 개수가 다릅니다.")
    _validate_compatibility(metadata)
    try:
        bundle = joblib.load(directory / "model.joblib")
    except Exception as exc:
        raise ArtifactValidationError("신뢰된 아티팩트의 model.joblib을 로드하지 못했습니다.") from exc
    required_bundle_keys = {"model_type", "mean", "std", "feature_order", "prediction_unit", "class_order"}
    if not isinstance(bundle, dict) or not required_bundle_keys.issubset(bundle):
        raise ArtifactValidationError("model.joblib의 예측 bundle 구조가 올바르지 않습니다.")
    if bundle["feature_order"] != feature_order or bundle["prediction_unit"] != metadata["prediction_unit"]:
        raise ArtifactValidationError("model bundle과 metadata/schema가 일치하지 않습니다.")
    smoke = pd.DataFrame([[0.0] * len(feature_order)], columns=feature_order)
    probability = predict_bundle_probabilities(bundle, smoke, schema)
    if probability.shape != (1,) or not np.isfinite(probability).all():
        raise ArtifactValidationError("아티팩트 소형 예측 smoke test가 실패했습니다.")
    return {"path": directory, "manifest": manifest, "metadata": metadata, "schema": schema, "metrics": metrics, "bundle": bundle}


def candidate_path(artifact_root: str | Path, artifact_id: str) -> Path:
    root = Path(artifact_root).resolve()
    return _trusted_path(root, root / "candidate" / _artifact_id(artifact_id))


def load_production_artifact(artifact_root: str | Path) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    current = root / "production" / "current"
    if not current.is_dir():
        raise ArtifactValidationError("승인된 production 아티팩트가 없습니다. 자동 재학습은 실행하지 않습니다.")
    return validate_artifact(root, current, expected_approval="production")


def _previous_destination(root: Path, artifact_id: str) -> Path:
    base = root / "previous" / _artifact_id(artifact_id)
    if not base.exists():
        return base
    return root / "previous" / f"{artifact_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def promote_candidate(artifact_root: str | Path, artifact_id: str) -> Path:
    root = Path(artifact_root).resolve()
    source = candidate_path(root, artifact_id)
    validate_artifact(root, source, expected_approval="candidate")
    production_root = root / "production"
    previous_root = root / "previous"
    production_root.mkdir(parents=True, exist_ok=True)
    previous_root.mkdir(parents=True, exist_ok=True)
    current = production_root / "current"
    previous_id = _read_json(current / "metadata.json").get("artifact_id") if current.exists() else None
    stage = root / f".promote-{uuid4().hex}"
    backup = _previous_destination(root, previous_id) if previous_id else None
    try:
        shutil.copytree(source, stage)
        _refresh_manifest(stage, "production", previous_id)
        validate_artifact(root, stage, expected_approval="production")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    moved_current = False
    try:
        if current.exists():
            os.replace(current, backup)
            moved_current = True
        os.replace(stage, current)
    except Exception:
        if moved_current and backup and backup.exists() and not current.exists():
            os.replace(backup, current)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return current


def rollback_production(artifact_root: str | Path) -> Path:
    root = Path(artifact_root).resolve()
    previous_root = root / "previous"
    candidates = [path for path in previous_root.iterdir() if path.is_dir()] if previous_root.exists() else []
    if not candidates:
        raise ArtifactValidationError("rollback 가능한 previous 아티팩트가 없습니다.")
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    validate_artifact(root, source, expected_approval="production")
    current = root / "production" / "current"
    current_id = _read_json(current / "metadata.json").get("artifact_id") if current.exists() else None
    stage = root / f".rollback-{uuid4().hex}"
    try:
        shutil.copytree(source, stage)
        _refresh_manifest(stage, "production", current_id)
        validate_artifact(root, stage, expected_approval="production")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    backup = _previous_destination(root, current_id) if current_id else None
    moved_current = False
    try:
        if current.exists():
            os.replace(current, backup)
            moved_current = True
        os.replace(stage, current)
        shutil.rmtree(source, ignore_errors=True)
    except Exception:
        if moved_current and backup and backup.exists() and not current.exists():
            os.replace(backup, current)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return current


def create_evaluation_candidate(
    artifact_root: str | Path,
    project_root: str | Path,
    prediction_bundle: dict[str, Any],
    payload: dict[str, Any],
    selected_metrics: dict[str, Any],
    production_gate_audit: dict[str, Any],
    bootstrap_rows: list[dict[str, Any]],
) -> Path:
    feature_order = list(payload["feature_columns"])
    available = list(prediction_bundle.get("available_feature_columns", feature_order))
    sparse_categories = [
        feature
        for feature in feature_order
        if re.fullmatch(r"(?:team|opponent)_(?:[A-Z]{2,4}|[가-힣]+)", feature)
    ]
    required_features = [
        feature for feature in feature_order if feature not in sparse_categories
    ]
    metadata = {
        "model_name": payload["selected_model"],
        "model_family": payload["model_type"],
        "selected_candidate": payload["selected_model"],
        "prediction_unit": payload["prediction_unit"],
        "training_start_year": payload.get("training_start_year"),
        "training_cutoff_date": payload["prediction_training_cutoff"],
        "target_name": "target_home_win" if payload["prediction_unit"] == "game" else "target_win",
        "feature_count": len(feature_order),
    }
    schema = {
        "feature_names": feature_order,
        "feature_order": feature_order,
        "feature_count": len(feature_order),
        "required_features": required_features,
        "optional_features": sorted(
            set(sparse_categories)
            | {feature for feature in available if feature not in feature_order}
        ),
        "expected_dtype": "float64",
        "missing_value_policy": "required numeric features must be present; absent one-hot team categories are filled with zero",
    }
    metrics = {
        "selected_candidate_metrics": selected_metrics,
        "confidence_metrics": payload.get("confidence_metrics"),
        "calibration_table": payload.get("calibration_table"),
        "high_confidence_backtest_summary": payload.get("high_confidence_backtest_summary"),
        "selected_model_probability_spread": payload.get("selected_model_probability_spread"),
        "bootstrap_results": bootstrap_rows,
        "production_gate": production_gate_audit,
    }
    return create_candidate_artifact(artifact_root, prediction_bundle, metadata, schema, metrics, project_root)
