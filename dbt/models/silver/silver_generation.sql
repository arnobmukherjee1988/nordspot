{{ config(order_by='(zone, valid_time)') }}

/*
    silver_generation
    ─────────────────
    Cleans and validates the raw generation_actual Silver table.

    Transforms applied:
      - Explicit column selection (no SELECT *)
      - Explicit DateTime cast on valid_time
      - Negative MW values filtered out (generation cannot be negative)
      - NULL rows on mandatory columns dropped
      - dbt_loaded_at audit column added for lineage tracking
*/

SELECT
    toDateTime(valid_time)  AS valid_time,
    zone,
    total_mw,
    wind_mw,
    solar_mw,
    now()                   AS dbt_loaded_at

FROM {{ source('nordspot', 'generation_actual') }}

WHERE valid_time IS NOT NULL
  AND zone      IS NOT NULL
  AND total_mw  >= 0
  AND wind_mw   >= 0
  AND solar_mw  >= 0
