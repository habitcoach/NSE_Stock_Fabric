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

# ### Stock Consolidation Logic
# 
# This notebook identifies stocks in potential consolidation or breakout phases using technical indicators.
# 
# RSI (Relative Strength Index):
# - RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
# - RSI ranges from 0 to 100.
# - RSI around 50 indicates consolidation (neither overbought nor oversold).
# - RSI > 70: Typically overbought (possible reversal or pullback).
# - RSI < 30: Typically oversold (possible bounce).
# - RSI = 100 - (100 / (1 + RS)) where RS (Relative Strength) =
# Average Gain (14 periods) ÷ Average Loss (14 periods)
# 
# MACD (Moving Average Convergence Divergence):
# - MACD Line = EMA(12-day) − EMA(26-day)
# - Signal Line = 9-day EMA of MACD Line
# - Histogram = MACD Line − Signal Line
# - Histogram > 0: Bullish momentum (MACD above Signal)
# - Histogram < 0: Bearish momentum (MACD below Signal)
# - Used to identify trend shifts and momentum.
# 
# Rate of Change (ROC)
# 
# ROC is a momentum indicator that measures the percentage change in price between the current close and the close n periods ago.
# 
# Why it matters:
# - Positive ROC = upward momentum
# - Negative ROC = downward momentum
# - Near 0 ROC = sideways movement / consolidation
# 
# The notebook computes RSI and MACD histogram values to detect stocks showing signs of trend continuation or potential breakout.
# 
# 
# What Are Bollinger Bands?
# 
# Bollinger Bands consist of three lines:
# 
# - Middle Band: A simple moving average (typically 20-day SMA)
# - Upper Band: Middle band + (2 × standard deviation)
# - Lower Band: Middle band − (2 × standard deviation)
# 
# - Upper Band = SMA𝑛+𝑘⋅𝜎𝑛
# - Lower Band=  SMA𝑛−𝑘⋅𝜎𝑛
# 
# Where:
# 
# n is the lookback period (commonly 20)
# k is the number of standard deviations (usually 2)
# σₙ is the rolling standard deviation of closing price
# 
# Why Use Bollinger Bands?
# 
# Volatility Detection: Bands expand when volatility increases and contract when it's low.
# 
# Overbought/Oversold Signals:
# - When price touches or crosses the upper band, it may be overbought.
# - When price touches or crosses the lower band, it may be oversold.
# - Breakout Strategy: When bands are very tight, a price breakout may be imminent.
# 
# 
# How Bollinger Bands Help With Mean Reversion
# 
# When price:
# - Touches or crosses the lower band → it may be oversold → price might rise back to the mean
# - Touches or crosses the upper band → it may be overbought → price might fall back to the mean
# 
# ⚠️ Note: Mean reversion works best in sideways (non-trending) markets, not during strong trends.
# 
# price_band_position - for bollinger band: 
# 
# price_band_position = (close - lower_band) / (upper_band - lower_band)
# 
# Intuition:
# - If the price is equal to the lower band, position = 0.0
# - If the price is equal to the upper band, position = 1.0
# - If the price is equal to the middle, position = 0.5
# - If it’s below the lower or above the upper band, the value can be (<0 or >1).
# 
# 
# 
# 
# 
# | Indicator       | Focus              | Key Usage                    |
# | --------------- | ------------------ | ---------------------------- |
# | RSI             | Momentum           | Overbought/Oversold zones    |
# | MACD            | Trend & Momentum   | Buy/Sell signals, crossovers |
# | ROC             | Price acceleration | % change over time           |
# | Bollinger Bands | Volatility         | Mean-reversion, breakouts    |
# 
# 
# ### Breakout Candidate Criteria
# 
# A stock is tagged as a breakout candidate if all of the following are true:
# 
# - ROC (14-day) is between 0% and 5%
# Reason: Price has started moving but hasn’t already hit the 3–5% target.
# 
# - Price Band Position is ≥ 0.8
# Reason: Current close is near the upper end of the Bollinger Bands — price is pressing against resistance.
# 
# - RSI (14) is ≥ 50
# Reason: Indicates bullish momentum (above neutral zone).
# 
# - MACD Histogram Label is "Positive (Bullish)"
# Reason: Confirms that trend momentum has shifted upward.
# 
# - Volume Ratio (5-day / 30-day) is ≥ 1.5
# Reason: Breakouts often happen with a surge in volume.


# CELL ********************

from pyspark.sql.functions import stddev  # for standard deviation
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
            "Authorization": "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3REFWOTMiLCJqdGkiOiI2ODk2ZDAyNmUxYzg2NTY1NzcwZWE3ODUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc1NDcxNDE1MCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzU0Nzc2ODAwfQ.pVTMdTx_CitFSacjU9z-P0cet_KbWZtktYevtP8YMgw"  # Replace with your actual token
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
        # display(df)

        # Step 3: Calculate Rolling Stats
        windowSpec_30 = Window.orderBy("datetime_ist").rowsBetween(-29, 0)
        windowSpec_5 = Window.orderBy("datetime_ist").rowsBetween(-4, 0)

        df = df.withColumn("rolling_high", spark_max("high").over(windowSpec_30))
        df = df.withColumn("rolling_low", spark_min("low").over(windowSpec_30))
        df = df.withColumn("rolling_avg_close", avg("close").over(windowSpec_30))
        df = df.withColumn("rolling_range_pct", ((col("rolling_high") - col("rolling_low")) / col("rolling_avg_close")) * 100)
        df = df.withColumn("rows_in_window", count("*").over(windowSpec_30))
        

        # Step 3.1: Calculate RSI (14-day)
        windowSpec_14 = Window.orderBy("datetime_ist").rowsBetween(-13, 0)

        # Step 3.1.1: Daily change in close
        df = df.withColumn("close_prev", lag("close").over(Window.orderBy("datetime_ist")))
        df = df.withColumn("change", col("close") - col("close_prev"))

        # Step 3.1.2: Separate gains and losses
        df = df.withColumn("gain", when(col("change") > 0, col("change")).otherwise(0.0))
        df = df.withColumn("loss", when(col("change") < 0, -col("change")).otherwise(0.0))

        # Step 3.1.3: Compute rolling average gain and loss over 14 periods
        df = df.withColumn("avg_gain", avg("gain").over(windowSpec_14))
        df = df.withColumn("avg_loss", avg("loss").over(windowSpec_14))

        # Step 3.1.4: Compute RSI
        df = df.withColumn("rs", when(col("avg_loss") == 0, lit(None)).otherwise(col("avg_gain") / col("avg_loss")))
        df = df.withColumn("rsi", when(col("rs").isNull(), 100.0).otherwise(100 - (100 / (1 + col("rs")))))

        # Step 3.2
        # ✅ New: Add avg volume & ratio
        df = df.withColumn("avg_volume_30", avg("volume").over(windowSpec_30))
        df = df.withColumn("avg_volume_5", avg("volume").over(windowSpec_5))
        df = df.withColumn("volume_ratio", col("avg_volume_5") / col("avg_volume_30"))
        #display(df)

        # Step 3.3: Calculate MACD Histogram Label (without retaining intermediate columns)
        short_ema_span = 12
        long_ema_span = 26
        signal_ema_span = 9

        short_ema_window = Window.orderBy("datetime_ist").rowsBetween(-short_ema_span + 1, 0)
        long_ema_window = Window.orderBy("datetime_ist").rowsBetween(-long_ema_span + 1, 0)
        signal_window = Window.orderBy("datetime_ist").rowsBetween(-signal_ema_span + 1, 0)

        df = df.withColumn("_ema_12", avg("close").over(short_ema_window))
        df = df.withColumn("_ema_26", avg("close").over(long_ema_window))
        df = df.withColumn("_macd", col("_ema_12") - col("_ema_26"))
        df = df.withColumn("_signal", avg("_macd").over(signal_window))
        df = df.withColumn("_hist", col("_macd") - col("_signal"))

        df = df.withColumn(
            "MACD_Histogram_Label",
            when(col("_hist") > 0, "Positive (Bullish)")
            .when(col("_hist") < 0, "Negative (Bearish)")
            .otherwise("Neutral (No momentum shift)")
        )

        # Drop intermediate columns
        df = df.drop("_ema_12", "_ema_26", "_macd", "_signal", "_hist")

        # Step 3.4: Calculate Rate of Change (ROC)
        roc_period = 14  # you can change this to any lookback period you prefer
        df = df.withColumn("close_n_days_ago", lag("close", roc_period).over(Window.orderBy("datetime_ist")))
        df = df.withColumn(
            "roc",
            when(col("close_n_days_ago").isNotNull(),
                 ((col("close") - col("close_n_days_ago")) / col("close_n_days_ago")) * 100
            ).otherwise(None)
        )

        # Step 3.5: Bollinger Bands and Price Position(30 days)
        df = df.withColumn("bollinger_mid", avg("close").over(windowSpec_30))
        df = df.withColumn("rolling_stddev", stddev("close").over(windowSpec_30))
        df = df.withColumn("bollinger_upper", col("bollinger_mid") + 2 * col("rolling_stddev"))
        df = df.withColumn("bollinger_lower", col("bollinger_mid") - 2 * col("rolling_stddev"))

        # Price Band Position
        df = df.withColumn(
            "price_band_position",
            when(
                (col("bollinger_upper") != col("bollinger_lower")),  # avoid division by zero
                (col("close") - col("bollinger_lower")) / (col("bollinger_upper") - col("bollinger_lower"))
            ).otherwise(None)
        )




        # Step 4: Filter for Consolidation
        consolidation_df = df.filter(
            (col("rolling_range_pct") <= 5) & (col("rows_in_window") == 30)
        )
        #display(df)

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

from datetime import datetime, timedelta
from pyspark.sql import DataFrame
from functools import reduce
from pyspark.sql.functions import round
import pytz  # ✅ For IST timezone support
import time  # ✅ For handling delay

# Step 1: Load stocks from the table
stock_list_df = spark.read.table("nse_symbol_inst")
print(f"Total stocks to process: {stock_list_df.count()}")

stock_tuples = [(row['instrument_key'], row['name']) for row in stock_list_df.collect()]

# Step 2: Set the date range dynamically based on IST
ist = pytz.timezone('Asia/Kolkata')
current_ist = datetime.now(ist)
stock_to_date = current_ist.strftime('%Y-%m-%d')
stock_from_date = (current_ist - timedelta(days=548)).strftime('%Y-%m-%d')

print(f"Date range set from {stock_from_date} to {stock_to_date}")

# Step 3: Process each stock with throttling
results = []
requests_per_min = 300
delay_seconds = 1  # made it one sec - earlier comment: 0.2 seconds delay between requests (original value: 60 / requests_per_min)

for index, (key, name) in enumerate(stock_tuples, start=1):
    print(f"Processing {index}/{len(stock_tuples)}: {key} ({name})")

    try:
        result_df = process_stock(key, name, stock_from_date, stock_to_date)

        if result_df is not None and isinstance(result_df, DataFrame):
            results.append(result_df)
    except Exception as e:
        print(f"[ERROR] Failed to process {key} ({name}): {e}")

    time.sleep(delay_seconds)  # ✅ Add delay to respect API rate limit

# Step 4: Combine all result DataFrames
if results:
    final_df = reduce(DataFrame.unionByName, results) #reduce individual df into final df

    finalzonestoc = final_df.select(
    "instrument_key",
    "stock_name",
    round("rolling_range_pct", 2).alias("rolling_range_pct_30days"),
    
    round("roc", 2).alias("roc"),
    # round("avg_volume_30", 2).alias("avg_volume_30d"),
    # round("avg_volume_5", 2).alias("avg_volume_5d"),
    round("rsi", 2).alias("rsi"),
    "MACD_Histogram_Label",
    round("volume_ratio", 2).alias("volume_ratio_5d/30d"),
    round("price_band_position", 2).alias("price_band_position"),
    "zone_id",
    "zone_end_date",
    #"rows_in_window"
).orderBy(col("zone_end_date").desc()) 
    # Get today's date in yyyyMMdd format
    today_date = datetime.today().strftime('%Y%m%d')

    # Add breakout candidate flag
    finalzonestocWithFlag = finalzonestoc.withColumn(
    "breakout_candidate",
    when(
        (col("roc") >= 0) & (col("roc") <= 5) &
        (col("price_band_position") >= 0.8) &
        (col("rsi") >= 50) &
        (col("MACD_Histogram_Label") == "Positive (Bullish)") &
        (col("volume_ratio_5d/30d") >= 1.5),
        "Yes"
    ).otherwise("No")
)

    # Create a dynamic table name with date suffix
    table_name = f"StockConTest_{today_date}"

    finalzonestocWithFlag.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable(table_name)

    display(finalzonestocWithFlag)
    print("✅ Successfully saved to table: StockConsolidation")
else:
    print("⚠️ No results to save.")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
