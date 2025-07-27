# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "92d28ad0-2830-4644-a41e-9ff49b3759e7",
# META       "default_lakehouse_name": "stklakehouse",
# META       "default_lakehouse_workspace_id": "5855c3a7-072f-4f70-a5d9-f028c9fc0a4e",
# META       "known_lakehouses": [
# META         {
# META           "id": "92d28ad0-2830-4644-a41e-9ff49b3759e7"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# ## In this code we are finding zones


# CELL ********************

import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from pyspark.sql.functions import col, to_timestamp, date_add

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("UpstoxStockAnalysis").getOrCreate()

# Step 2: Define correct Upstox API URL format
instrument_key = "NSE_EQ|INE585B01010"
unit = "days"
interval = "1"
to_date = "2025-07-26"
from_date = "2024-07-22"

url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"

headers = {
    "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3REFWOTMiLCJqdGkiOiI2ODgwNWQ0ZTY1ZjdiNjBkMDUxNmNhYTQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc1MzI0Mjk1OCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzUzMzA4MDAwfQ.6DfcautGz2fYafNRPyDvIWkWQCoZKhY2BMP7tYsBxTY"
}

# Step 3: Call the API
response = requests.get(url, headers=headers)
json_data = response.json()

# Step 4: Extract candles
candles = json_data["data"]["candles"]

# Step 5: Define schema
schema = StructType([
    StructField("datetime", StringType(), True),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("ignored", LongType(), True)
])

# Step 6: Create Spark DataFrame
df = spark.createDataFrame(candles, schema)

# Step 7: Adjust for IST (shift +1 day if needed)
df = df.withColumn("datetime", to_timestamp("datetime"))
df = df.withColumn("datetime_ist", date_add(col("datetime"), 1))

# Step 8: Display final result
display(df.select("datetime_ist", "open", "high", "low", "close", "volume"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark",
# META   "frozen": false,
# META   "editable": true
# META }

# CELL ********************

from pyspark.sql.window import Window
from pyspark.sql.functions import max as spark_max, min as spark_min, avg, count, col

# Define a 29-row rolling window (current row + 29 previous rows)
windowSpec = Window.orderBy("datetime_ist").rowsBetween(-29, 0)

# Add rolling metrics
df = df.withColumn("rolling_high", spark_max("high").over(windowSpec))
df = df.withColumn("rolling_low", spark_min("low").over(windowSpec))
df = df.withColumn("rolling_avg_close", avg("close").over(windowSpec))
df = df.withColumn("rolling_range_pct", ((col("rolling_high") - col("rolling_low")) / col("rolling_avg_close")) * 100)

# Add row count in the current rolling window
df = df.withColumn("rows_in_window", count("*").over(windowSpec))

# Display result
df02 = df.select("datetime_ist", "close", "rolling_high", "rolling_low", "rolling_avg_close", "rolling_range_pct", "rows_in_window")
display(df02)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col

# Filter where rolling range is <= 5 AND there are exactly 30 rows in the rolling window
consolidation_df = df.filter(
    (col("rolling_range_pct") <= 5) & (col("rows_in_window") == 30)
)

# Show the result
display(consolidation_df.select("datetime_ist", "close", "rolling_range_pct", "rows_in_window"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, datediff, when, sum as spark_sum, to_date

# Ensure datetime_ist is in date format (strip time part if present)
consolidation_df = consolidation_df.withColumn("date", to_date("datetime_ist"))

# Sort by date
consolidation_df = consolidation_df.orderBy("date")

# Create lagged date column
window_spec = Window.orderBy("date")
consolidation_df = consolidation_df.withColumn("prev_date", lag("date").over(window_spec))

# Identify where new zones start (gap > 3 day)
consolidation_df = consolidation_df.withColumn(
    "is_new_zone",
    when(col("prev_date").isNull() | (datediff(col("date"), col("prev_date")) > 3), 1).otherwise(0)
)

# Assign zone_id using cumulative sum
consolidation_df = consolidation_df.withColumn(
    "zone_id",
    spark_sum("is_new_zone").over(window_spec)
)

# Optional: Drop helper columns
consolidation_df = consolidation_df.drop("prev_date", "is_new_zone", "date")

# Show result
display(consolidation_df.select("datetime_ist", "close", "rolling_range_pct", "rows_in_window", "zone_id"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

# Define window to get the latest datetime_ist per zone
zone_window = Window.partitionBy("zone_id").orderBy(col("datetime_ist").desc())

# Add a rank column to identify the latest datetime in each zone
consolidation_df_ranked = consolidation_df.withColumn("rank", row_number().over(zone_window))

# Filter to keep only the latest row per zone
latest_datetime_per_zone = (
    consolidation_df_ranked
    .filter(col("rank") == 1)
    .drop("rank")
    .withColumn("zone_end_date", col("datetime_ist"))  # Add zone_end_date column
)

# Show result
display(latest_datetime_per_zone.select("zone_id", "zone_end_date", "close", "rolling_range_pct"))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
