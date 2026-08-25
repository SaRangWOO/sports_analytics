CREATE TABLE IF NOT EXISTS pitcher_game_logs (
    game_date date NOT NULL,
    game_id varchar(20) NOT NULL,
    team varchar(20) NOT NULL,
    opponent varchar(20) NOT NULL,
    home_away char(1) NOT NULL CHECK (home_away IN ('H', 'A')),
    pitcher_index smallint NOT NULL CHECK (pitcher_index > 0),
    pitcher_id integer,
    pitcher_name varchar(80) NOT NULL,
    is_starter boolean NOT NULL,
    entry varchar(20) NOT NULL,
    decision varchar(20),
    innings_outs smallint NOT NULL CHECK (innings_outs >= 0),
    batters_faced smallint,
    pitch_count smallint,
    at_bats smallint,
    hits_allowed smallint,
    home_runs_allowed smallint,
    walks_hbp smallint,
    strikeouts smallint,
    runs_allowed smallint,
    earned_runs smallint,
    game_era numeric(6, 2),
    collected_at timestamptz NOT NULL,
    data_source varchar(80) NOT NULL,
    PRIMARY KEY (game_id, team, pitcher_index)
);

CREATE INDEX IF NOT EXISTS pitcher_game_logs_date_team_idx
    ON pitcher_game_logs (game_date, team);
CREATE INDEX IF NOT EXISTS pitcher_game_logs_starter_idx
    ON pitcher_game_logs (pitcher_id, game_date) WHERE is_starter;

CREATE TABLE IF NOT EXISTS pregame_pitching_snapshots (
    snapshot_time timestamp NOT NULL,
    reference_date date NOT NULL,
    game_id varchar(40) NOT NULL,
    team varchar(20) NOT NULL,
    opponent varchar(20) NOT NULL,
    home_away char(1) NOT NULL CHECK (home_away IN ('H', 'A')),
    starter_name varchar(80),
    starter_source varchar(20) NOT NULL,
    starter_info_quality numeric(3, 2) NOT NULL,
    starter_era numeric(6, 2),
    starter_whip numeric(6, 2),
    bullpen_fatigue_label varchar(20),
    recent_3day_games smallint,
    data_source varchar(100),
    note text,
    PRIMARY KEY (reference_date, game_id, team, snapshot_time)
);

CREATE TABLE IF NOT EXISTS pregame_lineup_snapshots (
    snapshot_time timestamp NOT NULL,
    reference_date date NOT NULL,
    game_id varchar(40) NOT NULL,
    team varchar(20) NOT NULL,
    home_away char(1) NOT NULL CHECK (home_away IN ('H', 'A')),
    lineup_source varchar(20) NOT NULL,
    lineup_info_quality numeric(3, 2) NOT NULL,
    batting_order smallint NOT NULL,
    position varchar(20),
    player_name varchar(80) NOT NULL,
    war numeric(8, 3),
    data_source varchar(100),
    PRIMARY KEY (reference_date, game_id, team, snapshot_time, batting_order)
);

CREATE INDEX IF NOT EXISTS pregame_pitching_snapshot_asof_idx
    ON pregame_pitching_snapshots (reference_date, game_id, team, snapshot_time DESC);
CREATE INDEX IF NOT EXISTS pregame_lineup_snapshot_asof_idx
    ON pregame_lineup_snapshots (reference_date, game_id, team, snapshot_time DESC);
