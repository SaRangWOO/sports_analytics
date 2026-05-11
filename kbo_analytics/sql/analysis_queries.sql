-- KBO Team Performance Dashboard queries
-- Source tables:
--   game_results: weekly game-level results
--   player_game_stats: weekly player-level box score records

-- 1. Weekly team record
SELECT
    DATE_TRUNC('week', date::date)::date AS week_start,
    COUNT(*) AS games,
    SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'Loss' THEN 1 ELSE 0 END) AS losses,
    SUM(CASE WHEN result = 'Draw' THEN 1 ELSE 0 END) AS draws,
    ROUND(SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0), 3) AS win_rate,
    ROUND(AVG(score_team - score_opp), 2) AS avg_run_diff
FROM game_results
WHERE status = 'Final'
GROUP BY week_start
ORDER BY week_start;

-- 2. Opponent matchup record
SELECT
    opponent,
    COUNT(*) AS games,
    SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'Loss' THEN 1 ELSE 0 END) AS losses,
    SUM(CASE WHEN result = 'Draw' THEN 1 ELSE 0 END) AS draws,
    ROUND(SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0), 3) AS win_rate,
    SUM(score_team - score_opp) AS run_diff
FROM game_results
WHERE status = 'Final'
GROUP BY opponent
ORDER BY win_rate DESC, run_diff DESC;

-- 3. Home and away split
SELECT
    home_away,
    COUNT(*) AS games,
    SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'Loss' THEN 1 ELSE 0 END) AS losses,
    ROUND(SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0), 3) AS win_rate,
    ROUND(AVG(score_team), 2) AS avg_runs_scored,
    ROUND(AVG(score_opp), 2) AS avg_runs_allowed
FROM game_results
WHERE status = 'Final'
GROUP BY home_away
ORDER BY home_away;

-- 4. Monthly trend
SELECT
    DATE_TRUNC('month', date::date)::date AS month,
    COUNT(*) AS games,
    SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'Loss' THEN 1 ELSE 0 END) AS losses,
    ROUND(SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0), 3) AS win_rate,
    ROUND(AVG(score_team - score_opp), 2) AS avg_run_diff
FROM game_results
WHERE status = 'Final'
GROUP BY month
ORDER BY month;

-- 5. Top hitters by OPS proxy
SELECT
    player_name,
    team,
    COUNT(DISTINCT game_id) AS games,
    SUM(plate_appearances) AS pa,
    SUM(at_bats) AS ab,
    SUM(hits) AS hits,
    SUM(doubles) AS doubles,
    SUM(triples) AS triples,
    SUM(home_runs) AS home_runs,
    ROUND(SUM(hits)::numeric / NULLIF(SUM(at_bats), 0), 3) AS batting_avg,
    ROUND((SUM(hits) + SUM(walks))::numeric / NULLIF(SUM(plate_appearances), 0), 3) AS obp_proxy,
    ROUND((SUM(hits) + SUM(doubles) + 2 * SUM(triples) + 3 * SUM(home_runs))::numeric / NULLIF(SUM(at_bats), 0), 3) AS slg_proxy
FROM player_game_stats
WHERE plate_appearances > 0
GROUP BY player_name, team
HAVING SUM(plate_appearances) >= 20
ORDER BY slg_proxy DESC, obp_proxy DESC;

-- 6. Pitcher workload and run prevention
SELECT
    player_name,
    team,
    COUNT(DISTINCT game_id) AS games,
    SUM(innings_pitched) AS innings_pitched,
    SUM(pitches) AS pitches,
    ROUND((SUM(earned_runs) * 9.0 / NULLIF(SUM(innings_pitched), 0))::numeric, 2) AS era_proxy,
    ROUND((SUM(walks_allowed) + SUM(hits_allowed))::numeric / NULLIF(SUM(innings_pitched), 0), 2) AS whip_proxy,
    SUM(strikeouts_pitched) AS strikeouts
FROM player_game_stats
WHERE innings_pitched > 0
GROUP BY player_name, team
ORDER BY innings_pitched DESC, era_proxy ASC;
