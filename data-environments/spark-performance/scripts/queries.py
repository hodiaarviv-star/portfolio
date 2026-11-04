import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, desc
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import sum as spark_sum


# =========================================================
# Relative project paths
# =========================================================

root = os.getcwd()

data_dir = os.path.join(root, "data")

table_path = os.path.join(data_dir, "table.parquet")
partitioned_table_path = os.path.join(
    data_dir,
    "partitioned_table.parquet"
)


# =========================================================
# Choose the table to query
# First run: table_path
# Second run: partitioned_table_path
# =========================================================

joined_table_path = table_path


# =========================================================
# Create SparkSession
# =========================================================

spark = (
    SparkSession.builder
    .appName("Business Queries")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.driver.memory", "1g")
    .config("spark.memory.fraction", "0.7")
    .config("spark.memory.storageFraction", "0.6")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# Measure the total time:
# reading the table and running both queries
# =========================================================

start_time = time.perf_counter()

df = spark.read.parquet(joined_table_path)


# =========================================================
# Query 1:
# Five cities with the highest total sales
# =========================================================

query1 = (
    df.groupBy("city")
    .agg(
        spark_sum("sales_amount").alias("total_sales"),
        count("*").alias("number_of_orders")
    )
    .orderBy(desc("total_sales"))
    .limit(5)
)

print("\n================ QUERY 1 ================")
print("Five cities with the highest total sales:")
query1.show(truncate=False)


# =========================================================
# Query 2:
# Five cities with the highest average order amount
# =========================================================

query2 = (
    df.groupBy("city")
    .agg(
        spark_round(
            avg("sales_amount"),
            2
        ).alias("average_order_amount")
    )
    .orderBy(desc("average_order_amount"))
    .limit(5)
)

print("\n================ QUERY 2 ================")
print("Five cities with the highest average order amount:")
query2.show(truncate=False)


# =========================================================
# Total execution time of both queries
# =========================================================

total_time = time.perf_counter() - start_time

print("\n================ EXECUTION TIME ================")
print(f"Table path: {joined_table_path}")
print(f"Total execution time: {total_time:.2f} seconds")

spark.stop()