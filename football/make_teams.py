# make_teams.py
import copy
import os
import random
from itertools import combinations
from players import players as PLAYERS

HISTORY_FILE = "match_history.txt"

def load_history():
    history = {}
    if not os.path.exists(HISTORY_FILE): return history
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            pair = tuple(sorted(line.strip().split(",")))
            history[pair] = history.get(pair, 0) + 1
    return history

def save_history(team1, team2):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        for team in [team1, team2]:
            names = sorted([p["name"] for p in team])
            for combo in combinations(names, 2):
                f.write(f"{combo[0]},{combo[1]}\n")

def team_metrics(team):
    adj_skills, adj_staminas, injury_score = [], [], 0
    for p in team:
        s, st, i = p["skill"], p["stamina"], p.get("injury", 0)
        if i == 1: s *= 0.9; st *= 0.8; injury_score += 1
        elif i == 2: s *= 0.6; st *= 0.4; injury_score += 2
        adj_skills.append(s)
        adj_staminas.append(st)
    adj_skills.sort(reverse=True)
    return {
        "total_skill": sum(adj_skills), "total_stamina": sum(adj_staminas),
        "top2_skill": sum(adj_skills[:2]), "injury_score": injury_score
    }

def evaluate_balance(team1, team2, history_data):
    t1, t2 = team_metrics(team1), team_metrics(team2)
    # 인원수가 다를 경우(예: 5vs6)를 대비해 평균치 보정을 고려한 가중치
    size_diff = abs(len(team1) - len(team2))
    
    penalty = abs(t1["total_skill"] - t2["total_skill"]) * 10
    penalty += abs(t1["total_stamina"] - t2["total_stamina"]) * 4
    penalty += abs(t1["top2_skill"] - t2["top2_skill"]) * 8
    penalty += abs(t1["injury_score"] - t2["injury_score"]) * 5
    
    for team in [team1, team2]:
        for combo in combinations(sorted([p["name"] for p in team]), 2):
            penalty += history_data.get(combo, 0) * 15
    return penalty

def format_name(p):
    name = p["name"]
    injury = p.get("injury", 0)
    if injury == 1: return f"{name}(㊦)"
    if injury == 2: return f"{name}(㊤)"
    return name

def main_loop():
    while True:
        active_pool = copy.deepcopy(PLAYERS)
        history_data = load_history()

        print("\n" + "="*45)
        print(" [1단계] 오늘 안 온 사람 입력 (쉼표 구분)")
        absent_input = input("> ").strip()
        absent_names = [n.strip() for n in absent_input.split(",")] if absent_input else []
        
        # 1차 필터링
        temp_players = [p for p in active_pool if p["name"] not in absent_names]

        print("\n [2단계] 부상자 체크 (이름 입력 후 1~3단계)")
        injured_input = input("> ").strip()
        if injured_input:
            target_names = [n.strip() for n in injured_input.split(",")]
            for name in target_names:
                for p in temp_players:
                    if p["name"] == name:
                        print(f"[{name}] 상태: 1(경미)/2(심함)/3(제외)")
                        try: p["injury"] = int(input("선택: "))
                        except: p["injury"] = 0

        # 최종 가용 인원 (부상 3단계 제외)
        final_players = [p for p in temp_players if p.get("injury", 0) < 3]
        count = len(final_players)
        
        if count < 2:
            print("인원이 부족합니다.")
            continue

        # [핵심 로직] 인원 자동 배분 (휴식 없이 무조건 반반)
        t1_size = count // 2
        t2_size = count - t1_size # 홀수면 t2가 1명 더 많음
        
        print(f"\n총 {count}명 확인됨: {t1_size} vs {t2_size} 팀 빌딩을 시작합니다.")

        # 최적 조합 탐색
        all_combinations = list(combinations(range(count), t1_size))
        random.shuffle(all_combinations)
        
        best_score, best_match = float('inf'), None
        sample_size = min(len(all_combinations), 3000)
        
        for i in range(sample_size):
            t1_idx = all_combinations[i]
            t1 = [final_players[j] for j in t1_idx]
            t2 = [final_players[j] for j in range(count) if j not in t1_idx]
            
            score = evaluate_balance(t1, t2, history_data)
            if score < best_score:
                best_score, best_match = score, (t1, t2)

        if best_match:
            t1, t2 = best_match
            print("\n" + "★"*40)
            print(f"■ A팀 ({len(t1)}명): {', '.join([format_name(p) for p in t1])}")
            print(f"■ B팀 ({len(t2)}명): {', '.join([format_name(p) for p in t2])}")
            print("★"*40)
            
            confirm = input("\n기록하고 종료(y) / 다시 짜기(n) / 그냥 종료(q): ").lower()
            if confirm == 'y':
                save_history(t1, t2)
                print("저장되었습니다.")
                break
            elif confirm == 'q': break
            print("\n다시 설정을 시작합니다...")

if __name__ == "__main__":
    main_loop()