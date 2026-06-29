{{ config(order_by='(zone, valid_time)') }}

/*
    silver_load
    -----------
    Cleans and validates the raw load_actual Silver table.

    Transforms applied:
      - Explicit column selection
      - Explicit DateTime cast on valid_time
      - Zero and negative load values filtered (load must be positive)
      - NULL rows on mandatory columns dropped
      - dbt_loaded_at audit column added for lineage tracking
*/

SELECT
    toDateTime(valid_time)  AS valid_time,
    zone,
    value_mw,
    now()                   AS dbt_loaded_at

FROM {{ source('nordspot', 'load_actual') }}

WHERE valid_time IS NOT NULL
  AND zone       IS NOT NULL
  AND value_mw   > 0
