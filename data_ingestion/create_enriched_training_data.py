import os

from google.cloud import bigquery

KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/Users/AVM/AVM/gcp_key.json")
PROJECT_ID = os.getenv("AVM_PROJECT_ID", "avm-ml-485715")
DATASET = os.getenv("AVM_DATASET", "avm_dataset")
SOURCE_TABLE = os.getenv("AVM_SOURCE_TABLE", f"`{PROJECT_ID}.{DATASET}.listings_cleaned`")
TARGET_TABLE = os.getenv("AVM_TARGET_TABLE", f"`{PROJECT_ID}.{DATASET}.master_training_data_v4`")
LISTING_GEOCODES_TABLE = os.getenv("AVM_LISTING_GEOCODES_TABLE", f"`{PROJECT_ID}.{DATASET}.listing_geocodes`")


if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

client = bigquery.Client(project=PROJECT_ID)

query = f"""
CREATE OR REPLACE TABLE {TARGET_TABLE} AS
WITH
listings AS (
    SELECT
        *,
        SAFE_CAST(price AS FLOAT64) AS price_numeric,
        SAFE_CAST(bedrooms AS INT64) AS bedrooms_int,
        REPLACE(UPPER(REGEXP_EXTRACT(CONCAT(COALESCE(address, ''), ' ', COALESCE(description, '')), r'([A-Za-z]{{1,2}}\\d{{1,2}}\\s?\\d[A-Za-z]{{2}})')), ' ', '') AS postcode_clean,

        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'modern|renovated|refurbished|immaculate|luxury|stunning|contemporary') THEN 1 ELSE 0 END AS is_modern,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'needs work|fixer|tlc|modernisation|potential|project|dated|tired') THEN 1 ELSE 0 END AS needs_work,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'garden|patio|terrace|outdoor space|balcony') THEN 1 ELSE 0 END AS has_garden,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'garage|parking|driveway|off-street|drive') THEN 1 ELSE 0 END AS has_parking,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'view|views|overlook|panoramic|countryside|sea view') THEN 1 ELSE 0 END AS has_views,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'quiet|peaceful|secluded|cul-de-sac') THEN 1 ELSE 0 END AS is_quiet_location,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'station|transport|tube|train|underground') THEN 1 ELSE 0 END AS near_transport,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'school|schools|catchment') THEN 1 ELSE 0 END AS near_schools,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'period|victorian|edwardian|georgian|character|original features') THEN 1 ELSE 0 END AS is_period_property,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'new build|newly built|brand new|never lived') THEN 1 ELSE 0 END AS is_new_build,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'investment|tenant|let|rental income|buy to let') THEN 1 ELSE 0 END AS is_investment
    FROM {SOURCE_TABLE}
    WHERE SAFE_CAST(price AS FLOAT64) IS NOT NULL
      AND SAFE_CAST(price AS FLOAT64) BETWEEN 50000 AND 4000000
),

geocodes AS (
    SELECT
        REPLACE(UPPER(postcode), ' ', '') AS postcode_clean,
        AVG(SAFE_CAST(lat AS FLOAT64)) AS geocode_lat,
        AVG(SAFE_CAST(lon AS FLOAT64)) AS geocode_lon,
        ANY_VALUE(district) AS district
    FROM {LISTING_GEOCODES_TABLE}
    WHERE SAFE_CAST(lat AS FLOAT64) IS NOT NULL
      AND SAFE_CAST(lon AS FLOAT64) IS NOT NULL
    GROUP BY 1
),

epc_stats AS (
    SELECT
        REPLACE(UPPER(POSTCODE), ' ', '') AS postcode_clean,
        AVG(SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64)) AS avg_floor_area,
        AVG(SAFE_CAST(CURRENT_ENERGY_EFFICIENCY AS FLOAT64)) AS avg_energy_efficiency,
        APPROX_QUANTILES(SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64), 2)[OFFSET(1)] AS median_floor_area,
        COUNT(*) AS epc_records
    FROM `{PROJECT_ID}.{DATASET}.epc_raw`
    WHERE SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64) > 0
    GROUP BY 1
),

sold_homes AS (
    SELECT
        SAFE_CAST(lr.price AS FLOAT64) AS sold_price,
        REPLACE(UPPER(REGEXP_EXTRACT(lr.postcode, r'([A-Za-z]{{1,2}}\\d{{1,2}}\\s?\\d[A-Za-z]{{2}})')), ' ', '') AS postcode_clean,
        REPLACE(UPPER(REGEXP_EXTRACT(lr.postcode, r'([A-Za-z]{{1,2}}\\d{{1,2}}\\s?\\d)')), ' ', '') AS postcode_sector,
        UPPER(lr.town_city) AS town_city
    FROM `{PROJECT_ID}.{DATASET}.land_registry_raw` lr
    WHERE lr.transfer_date >= '2020-01-01'
      AND lr.price BETWEEN 50000 AND 5000000
      AND lr.postcode IS NOT NULL
),

sold_homes_spatial AS (
    SELECT
        s.sold_price,
        ST_GEOGPOINT(SAFE_CAST(g.lon AS FLOAT64), SAFE_CAST(g.lat AS FLOAT64)) AS sold_geom
    FROM sold_homes s
    JOIN `{PROJECT_ID}.{DATASET}.geocodes` g
      ON s.postcode_clean = REPLACE(UPPER(g.postcode), ' ', '')
    WHERE SAFE_CAST(g.lat AS FLOAT64) IS NOT NULL
      AND SAFE_CAST(g.lon AS FLOAT64) IS NOT NULL
),

sector_stats AS (
    SELECT
        postcode_sector,
        COUNT(*) AS sector_sales_count,
        AVG(sold_price) AS sector_avg_price,
        APPROX_QUANTILES(sold_price, 2)[OFFSET(1)] AS sector_median_price,
        MIN(sold_price) AS sector_min_price,
        MAX(sold_price) AS sector_max_price,
        STDDEV(sold_price) AS sector_price_stddev
    FROM sold_homes
    WHERE postcode_sector IS NOT NULL
    GROUP BY postcode_sector
    HAVING COUNT(*) >= 3
),

city_stats AS (
    SELECT
        town_city,
        COUNT(*) AS city_sales_count,
        AVG(sold_price) AS city_avg_price,
        APPROX_QUANTILES(sold_price, 2)[OFFSET(1)] AS city_median_price
    FROM sold_homes
    WHERE town_city IS NOT NULL
    GROUP BY town_city
    HAVING COUNT(*) >= 3
),

listing_geocoded AS (
    SELECT
        l.*,
        g.geocode_lat AS latitude_final,
        g.geocode_lon AS longitude_final,
        g.district,
        ST_GEOGPOINT(g.geocode_lon, g.geocode_lat) AS listing_geom
    FROM listings l
    LEFT JOIN geocodes g
      ON l.postcode_clean = g.postcode_clean
),

comps_ranked AS (
    SELECT
        l.main_image_url,
        s.sold_price,
        ROW_NUMBER() OVER (
            PARTITION BY l.main_image_url
            ORDER BY ST_DISTANCE(l.listing_geom, s.sold_geom)
        ) AS comp_rank
    FROM listing_geocoded l
    JOIN sold_homes_spatial s
      ON ST_DWITHIN(l.listing_geom, s.sold_geom, 2000)
),

comps_agg AS (
    SELECT
        main_image_url,
        AVG(sold_price) AS avg_nearby_sold_price,
        COUNT(*) AS comp_count
    FROM comps_ranked
    WHERE comp_rank <= 5
    GROUP BY main_image_url
)

SELECT
    l.price_numeric AS price,
    l.bedrooms_int AS bedrooms,
    l.property_type,
    l.city,
    l.latitude_final AS latitude,
    l.longitude_final AS longitude,
    l.district,

    l.is_modern,
    l.needs_work,
    l.has_garden,
    l.has_parking,
    l.has_views,
    l.is_quiet_location,
    l.near_transport,
    l.near_schools,
    l.is_period_property,
    l.is_new_build,
    l.is_investment,

    COALESCE(e.avg_floor_area, e.median_floor_area, 70) AS floor_area,
    COALESCE(e.avg_energy_efficiency, 60) AS energy_efficiency,
    COALESCE(e.epc_records, 0) AS epc_records,

    s.sector_sales_count,
    s.sector_avg_price,
    s.sector_median_price,
    s.sector_min_price,
    s.sector_max_price,
    s.sector_price_stddev,
    c.city_sales_count,
    c.city_avg_price,
    c.city_median_price,
    COALESCE(ca.avg_nearby_sold_price, s.sector_median_price, c.city_median_price) AS best_comp_price,
    COALESCE(ca.comp_count, 0) AS comp_count,
    CASE
        WHEN ca.avg_nearby_sold_price IS NOT NULL THEN 'spatial'
        WHEN s.sector_median_price IS NOT NULL THEN 'sector'
        WHEN c.city_median_price IS NOT NULL THEN 'city'
        ELSE 'none'
    END AS comp_source,

    COALESCE(l.main_image_url, '') AS main_image_url,
    l.address,
    l.description,

    l.listing_geom
FROM listing_geocoded l
LEFT JOIN epc_stats e
  ON l.postcode_clean = e.postcode_clean
LEFT JOIN sector_stats s
  ON REGEXP_EXTRACT(l.postcode_clean, r'([A-Za-z]{{1,2}}\\d{{1,2}}\\s?\\d)') = s.postcode_sector
LEFT JOIN city_stats c
  ON UPPER(l.city) = c.town_city
LEFT JOIN comps_agg ca
  ON l.main_image_url = ca.main_image_url
"""

print("🔧 Creating enriched training data...")
job = client.query(query)
job.result()

table = client.get_table(f"{PROJECT_ID}.{DATASET}.master_training_data_v4")
print(f"✅ Created table with {table.num_rows:,} rows and {len(table.schema)} columns")
