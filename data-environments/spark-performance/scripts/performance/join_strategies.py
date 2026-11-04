import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast


# =========================================================
# Relative project paths
# =========================================================

root = os.getcwd()
data_dir = os.path.join(root, "data")
cleaned_dir = os.path.join(data_dir, "cleaned")

orders_path = os.path.join(cleaned_dir, "zomato_orders")
restaurant_path = os.path.join(cleaned_dir, "zomato_restaurant")


# =========================================================
# Helper function: calculate Parquet directory size in MB
# =========================================================

def directory_size_mb(path):
    total_bytes = 0

    for current_path, directories, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(current_path, file_name)
            total_bytes += os.path.getsize(file_path)

    return total_bytes / (1024 * 1024)


# =========================================================
# Create SparkSession
# =========================================================

spark = (
    SparkSession.builder
    .appName("Join Strategies")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# Read the two base tables
# =========================================================

orders = spark.read.parquet(orders_path)
restaurant = spark.read.parquet(restaurant_path)

# The join columns must have the same data type
orders = orders.withColumn("r_id_long", col("r_id").cast("long"))


# =========================================================
# Display table sizes
# =========================================================

orders_size = directory_size_mb(orders_path)
restaurant_size = directory_size_mb(restaurant_path)

print("\n================ TABLE SIZES ================")
print(f"zomato_orders size: {orders_size:.2f} MB")
print(f"zomato_restaurant size: {restaurant_size:.2f} MB")


# =========================================================
# 1. Sort Merge Join
# =========================================================

print("\n\n================ SORT MERGE JOIN ================")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.join.preferSortMergeJoin", "true")
spark.catalog.clearCache()

start_time = time.perf_counter()

sort_merge_join = (
    orders.hint("merge")
    .join(
        restaurant.hint("merge"),
        orders["r_id_long"] == restaurant["id"],
        "inner"
    )
)

sort_merge_rows = sort_merge_join.count()

sort_merge_time = time.perf_counter() - start_time

print(f"Number of joined rows: {sort_merge_rows}")
print(f"Sort Merge Join time: {sort_merge_time:.2f} seconds")

print("\nPhysical plan:")
sort_merge_join.explain(mode="formatted")


# =========================================================
# Display the first five rows of the joined table
# =========================================================

print("\nFirst five rows of the joined table:")
sort_merge_join.show(5, truncate=False)


# =========================================================
# 2. Broadcast Hash Join
# zomato_orders (1.32MB) is smaller than zomato_restaurant
# (3.32MB), so orders is broadcast
# =========================================================

print("\n\n================ BROADCAST HASH JOIN ================")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))
spark.catalog.clearCache()

start_time = time.perf_counter()

broadcast_join = broadcast(orders).join(
    restaurant,
    orders["r_id_long"] == restaurant["id"],
    "inner"
)

broadcast_rows = broadcast_join.count()

broadcast_time = time.perf_counter() - start_time

print(f"Number of joined rows: {broadcast_rows}")
print(f"Broadcast Hash Join time: {broadcast_time:.2f} seconds")

print("\nPhysical plan:")
broadcast_join.explain(mode="formatted")


# =========================================================
# 3. Shuffle Hash Join
# zomato_orders (1.32MB) is smaller than zomato_restaurant
# (3.32MB), so orders is used as the build side
# =========================================================

print("\n\n================ SHUFFLE HASH JOIN ================")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.join.preferSortMergeJoin", "false")
spark.catalog.clearCache()

start_time = time.perf_counter()

shuffle_hash_join = orders.hint("shuffle_hash").join(
    restaurant,
    orders["r_id_long"] == restaurant["id"],
    "inner"
)

shuffle_hash_rows = shuffle_hash_join.count()

shuffle_hash_time = time.perf_counter() - start_time

print(f"Number of joined rows: {shuffle_hash_rows}")
print(f"Shuffle Hash Join time: {shuffle_hash_time:.2f} seconds")

print("\nPhysical plan:")
shuffle_hash_join.explain(mode="formatted")


# =========================================================
# Summary
# =========================================================

print("\n\n================ TIME SUMMARY ================")
print(f"Sort Merge Join:    {sort_merge_time:.2f} seconds")
print(f"Broadcast Hash Join: {broadcast_time:.2f} seconds")
print(f"Shuffle Hash Join:  {shuffle_hash_time:.2f} seconds")

times = {
    "Sort Merge Join": sort_merge_time,
    "Broadcast Hash Join": broadcast_time,
    "Shuffle Hash Join": shuffle_hash_time
}

fastest_strategy = min(times, key=times.get)

print(f"\nFastest strategy: {fastest_strategy}")
print(f"Fastest time: {times[fastest_strategy]:.2f} seconds")

spark.stop()