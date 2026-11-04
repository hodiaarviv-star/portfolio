import os
import time

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, desc
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import sum as spark_sum


# =========================================================
# Relative project paths
# =========================================================

root = os.getcwd()

data_dir = os.path.join(root, "data")

joined_table_path = os.path.join(data_dir, "table.parquet")


# =========================================================
# Choose the storage level for the current run
# Change these two lines before each run
# =========================================================

selected_name = "MEMORY_ONLY"
selected_level = StorageLevel.MEMORY_ONLY

# For testing MEMORY_AND_DISK, replace the two lines above with:
# selected_name = "MEMORY_AND_DISK"
# selected_level = StorageLevel.MEMORY_AND_DISK


# =========================================================
# Create SparkSession
# =========================================================

spark = (
    SparkSession.builder
    .appName("Storage Level Performance Test")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# Measure total execution time
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


# =========================================================
# Persist the source DataFrame between the two queries
# =========================================================

df.persist(storageLevel=selected_level)


# =========================================================
# Execute Query 1
# =========================================================

print(f"\n================ {selected_name} ================")

print("\nQuery 1 – five cities with the highest total sales:")
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

print("\nQuery 2 – five cities with the highest average order amount:")
query2.show(truncate=False)


# =========================================================
# Execution time
# =========================================================

execution_time = time.perf_counter() - start_time

print("\n================ RESULT ================")
print(f"Storage level: {selected_name}")
print(f"Execution time: {execution_time:.2f} seconds")


# =========================================================
# Clean up
# =========================================================

df.unpersist(blocking=True)
spark.stop()