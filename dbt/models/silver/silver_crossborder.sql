{{ config(order_by='(from_zone, to_zone, valid_time)') }}

/*
    silver_crossborder
    ------------------
    Cleans and validates the raw crossborder_flows Silver table.

    Transforms applied:
      - Explicit column selection
      - Explicit DateTime cast on valid_time
      - NULL rows on mandatory columns dropped
      - dbt_loaded_at audit column added for lineage tracking

    Note: value_mw can be negative (net import direction), so no sign filter here.
*/

SELECT
    toDateTime(valid_time)  AS valid_time,
    from_zone,
    to_zone,
    value_mw,
    now()                   AS dbt_loaded_at

FROM {{ source('nordspot', 'crossborder_flows') }}

WHERE valid_time IS NOT NULL
  AND from_zone  IS NOT NULL
  AND to_zone    IS NOT NULL
