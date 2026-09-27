-- Before the Call: reusable analytical views
--
-- The source table is created by src/18_build_sql_analytics.py.
-- Test results are descriptive and must not be used to retune the model.


DROP VIEW IF EXISTS vw_split_summary;

CREATE VIEW vw_split_summary AS
SELECT
    dataset_split,
    COUNT(*) AS observations,
    SUM(subscribed) AS subscribers,
    COUNT(*) - SUM(subscribed) AS non_subscribers,
    AVG(subscribed * 1.0) AS response_rate
FROM campaign_observations
GROUP BY dataset_split;


DROP VIEW IF EXISTS vw_test_score_deciles;

CREATE VIEW vw_test_score_deciles AS
WITH ranked AS (
    SELECT
        observation_id,
        subscribed,
        calibrated_probability,
        NTILE(10) OVER (
            ORDER BY calibrated_probability DESC,
                     observation_id
        ) AS score_decile
    FROM campaign_observations
    WHERE dataset_split = 'test'
)
SELECT
    score_decile,
    COUNT(*) AS observations,
    SUM(subscribed) AS subscribers,
    AVG(subscribed * 1.0) AS observed_response_rate,
    AVG(calibrated_probability) AS mean_predicted_probability,
    MIN(calibrated_probability) AS minimum_probability,
    MAX(calibrated_probability) AS maximum_probability
FROM ranked
GROUP BY score_decile;


DROP VIEW IF EXISTS vw_test_capacity_summary;

CREATE VIEW vw_test_capacity_summary AS
WITH capacities(capacity) AS (
    VALUES (0.10), (0.20)
),
ranked AS (
    SELECT
        observation_id,
        subscribed,
        calibrated_probability,
        ROW_NUMBER() OVER (
            ORDER BY calibrated_probability DESC,
                     observation_id
        ) AS score_rank,
        COUNT(*) OVER () AS total_observations,
        SUM(subscribed) OVER () AS total_subscribers
    FROM campaign_observations
    WHERE dataset_split = 'test'
),
capacity_assignments AS (
    SELECT
        capacities.capacity,
        ranked.*,
        CAST(
            ranked.total_observations
            * capacities.capacity
            + 0.999999
            AS INTEGER
        ) AS selected_limit
    FROM ranked
    CROSS JOIN capacities
)
SELECT
    capacity,
    MAX(total_observations) AS test_observations,
    MAX(total_subscribers) AS test_subscribers,
    MAX(selected_limit) AS selected_observations,
    SUM(
        CASE
            WHEN score_rank <= selected_limit
            THEN subscribed
            ELSE 0
        END
    ) AS selected_subscribers,
    SUM(
        CASE
            WHEN score_rank <= selected_limit
                 AND subscribed = 0
            THEN 1
            ELSE 0
        END
    ) AS false_prioritizations,
    SUM(
        CASE
            WHEN score_rank > selected_limit
                 AND subscribed = 1
            THEN 1
            ELSE 0
        END
    ) AS missed_subscribers,
    (
        SUM(
            CASE
                WHEN score_rank <= selected_limit
                THEN subscribed
                ELSE 0
            END
        ) * 1.0
        / MAX(selected_limit)
    ) AS precision_at_capacity,
    (
        SUM(
            CASE
                WHEN score_rank <= selected_limit
                THEN subscribed
                ELSE 0
            END
        ) * 1.0
        / MAX(total_subscribers)
    ) AS recall_at_capacity,
    (
        (
            SUM(
                CASE
                    WHEN score_rank <= selected_limit
                    THEN subscribed
                    ELSE 0
                END
            ) * 1.0
            / MAX(selected_limit)
        )
        /
        (
            MAX(total_subscribers) * 1.0
            / MAX(total_observations)
        )
    ) AS lift_at_capacity
FROM capacity_assignments
GROUP BY capacity;


DROP VIEW IF EXISTS vw_test_contact_audit;

CREATE VIEW vw_test_contact_audit AS
SELECT
    contact,
    COUNT(*) AS observations,
    SUM(subscribed) AS subscribers,
    AVG(subscribed * 1.0) AS observed_response_rate,
    AVG(calibrated_probability) AS mean_predicted_probability,
    AVG(calibrated_probability)
        - AVG(subscribed * 1.0) AS calibration_gap,
    SUM(selected_20) AS selected_at_20,
    AVG(selected_20 * 1.0) AS selection_rate_at_20,
    (
        SUM(
            CASE
                WHEN selected_20 = 1
                THEN subscribed
                ELSE 0
            END
        ) * 1.0
        / NULLIF(SUM(selected_20), 0)
    ) AS precision_at_20
FROM campaign_observations
WHERE dataset_split = 'test'
GROUP BY contact;


DROP VIEW IF EXISTS vw_test_month_audit;

CREATE VIEW vw_test_month_audit AS
SELECT
    month,
    COUNT(*) AS observations,
    SUM(subscribed) AS subscribers,
    AVG(subscribed * 1.0) AS observed_response_rate,
    AVG(calibrated_probability) AS mean_predicted_probability,
    AVG(calibrated_probability)
        - AVG(subscribed * 1.0) AS calibration_gap,
    SUM(selected_10) AS selected_at_10,
    SUM(selected_20) AS selected_at_20
FROM campaign_observations
WHERE dataset_split = 'test'
GROUP BY month;