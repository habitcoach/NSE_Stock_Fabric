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

# ### Stock Consolidation logic

# CELL ********************

import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
from pyspark.sql.functions import (
    col, to_timestamp, date_add, avg, count, max as spark_max, min as spark_min,
    lag, datediff, when, sum as spark_sum, to_date, row_number, lit
)
from pyspark.sql.window import Window

# Create Spark session (if not already created)
spark = SparkSession.builder.getOrCreate()

# Define the main function to process one stock
def process_stock(instrument_key: str, stock_name: str, stock_from_date: str, stock_to_date: str):
    try:
        # Step 1: Call the Upstox API
        url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/days/1/{stock_to_date}/{stock_from_date}"
        headers = {
            "Authorization": "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3REFWOTMiLCJqdGkiOiI2ODg1YzUwN2JiZWI1ODQ1YWMzYzdlOWUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc1MzU5NzE5MSwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzUzNjUzNjAwfQ.1EDIl_vwFBg0jeLX40X70s7Hb1-yO6u0phCwBQIFo3Y"  # Replace with your actual token
        }
        response = requests.get(url, headers=headers)
        json_data = response.json()
        candles = json_data["data"]["candles"]

        # Step 2: Create DataFrame
        schema = StructType([
            StructField("datetime", StringType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", LongType(), True),
            StructField("ignored", LongType(), True)
        ])
        df = spark.createDataFrame(candles, schema)
        df = df.withColumn("datetime", to_timestamp("datetime"))
        df = df.withColumn("datetime_ist", date_add(col("datetime"), 1))

        # Step 3: Calculate Rolling Stats
        windowSpec = Window.orderBy("datetime_ist").rowsBetween(-29, 0)
        df = df.withColumn("rolling_high", spark_max("high").over(windowSpec))
        df = df.withColumn("rolling_low", spark_min("low").over(windowSpec))
        df = df.withColumn("rolling_avg_close", avg("close").over(windowSpec))
        df = df.withColumn("rolling_range_pct", ((col("rolling_high") - col("rolling_low")) / col("rolling_avg_close")) * 100)
        df = df.withColumn("rows_in_window", count("*").over(windowSpec))

        # Step 4: Filter for Consolidation
        consolidation_df = df.filter(
            (col("rolling_range_pct") <= 5) & (col("rows_in_window") == 30)
        )
        display(consolidation_df)

        # Step 5: Detect Zone Changes
        consolidation_df = consolidation_df.withColumn("date", to_date("datetime_ist"))
        consolidation_df = consolidation_df.orderBy("date")
        window_spec = Window.orderBy("date")
        consolidation_df = consolidation_df.withColumn("prev_date", lag("date").over(window_spec))
        consolidation_df = consolidation_df.withColumn(
            "is_new_zone",
            when(col("prev_date").isNull() | (datediff(col("date"), col("prev_date")) > 3), 1).otherwise(0)
        )
        consolidation_df = consolidation_df.withColumn("zone_id", spark_sum("is_new_zone").over(window_spec))
        consolidation_df = consolidation_df.drop("prev_date", "is_new_zone", "date")

        # Step 6: Get Latest Date per Zone
        zone_window = Window.partitionBy("zone_id").orderBy(col("datetime_ist").desc())
        consolidation_df_ranked = consolidation_df.withColumn("rank", row_number().over(zone_window))
        latest_df = (
            consolidation_df_ranked
            .filter(col("rank") == 1)
            .drop("rank")
            .withColumn("zone_end_date", col("datetime_ist"))
            .withColumn("instrument_key", lit(instrument_key))
            .withColumn("stock_name", lit(stock_name))
        )

        return latest_df

    except Exception as e:
        print(f"[ERROR] Failed to process {instrument_key}: {e}")
        return None


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from datetime import datetime
from pyspark.sql import DataFrame
from functools import reduce
from pyspark.sql.functions import round
import time  # ✅ For handling delay

# Step 1: Load stocks from the table
stock_list_df = spark.read.table("nse_symbol_inst").limit(50)
print(f"Total stocks to process: {stock_list_df.count()}")

stock_tuples = [(row['instrument_key'], row['name']) for row in stock_list_df.collect()]

# Step 2: Set the date range
stock_from_date = "2024-07-22"
stock_to_date = "2025-07-26"

# Step 3: Process each stock with throttling
results = []
requests_per_min = 300
delay_seconds = 1  # made it one sec - earlier comment: 0.2 seconds delay between requests (orginal value: 60 / requests_per_min )

for index, (key, name) in enumerate(stock_tuples, start=1):
    print(f"Processing {index}/{len(stock_tuples)}: {key} ({name})")

    try:
        result_df = process_stock(key, name, stock_from_date, stock_to_date)
        display(result_df)

        if result_df is not None and isinstance(result_df, DataFrame):
            results.append(result_df)
    except Exception as e:
        print(f"[ERROR] Failed to process {key} ({name}): {e}")

    time.sleep(1)  # ✅ Add delay to respect API rate limit

# Step 4: Combine all result DataFrames
if results:
    final_df = reduce(DataFrame.unionByName, results)

    finalzonestoc = final_df.select(
        "instrument_key",
        "stock_name",
        round("rolling_range_pct", 2).alias("rolling_range_pct"),  # ✅ Rounded to 2 decimal places
        "zone_id",
        "zone_end_date",
        "rows_in_window"
    )

    finalzonestoc.write.mode("overwrite").saveAsTable("StockConsolidation")
    display(finalzonestoc)
    print("✅ Successfully saved to table: StockConsolidation")
else:
    print("⚠️ No results to save.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
