import os

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

table_path = os.path.join(data_dir, "table.parquet")
partitioned_table_path = os.path.join(
    data_dir,
    "partitioned_table.parquet"
)


# =========================================================
# Create SparkSession
# =========================================================

spark = (
    SparkSession.builder
    .appName("Write Joined Tables")
    .config("spark.sql.adaptive.enabled", "false")
    .config(
        "spark.sql.autoBroadcastJoinThreshold",
        str(100 * 1024 * 1024)
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# Read the base tables
# =========================================================

orders = spark.read.parquet(orders_path)
restaurant = spark.read.parquet(restaurant_path)

# Match the data types of the join columns
orders = orders.withColumn(
    "r_id_long",
    col("r_id").cast("long")
)


# =========================================================
# Join using Broadcast Hash Join
# zomato_orders (1.32MB) is smaller than zomato_restaurant
# (3.32MB), so orders is the side that gets broadcast
# =========================================================

joined_table = broadcast(orders).join(
    restaurant,
    orders["r_id_long"] == restaurant["id"],
    "inner"
)


# =========================================================
# Verify the joined table
# =========================================================

print("\nFirst five rows of the joined table:")
joined_table.show(5, truncate=False)

print(f"\nNumber of joined rows: {joined_table.count()}")


# =========================================================
# Write the regular Parquet table
# =========================================================

joined_table.write \
    .mode("overwrite") \
    .parquet(table_path)

print(f"\nRegular table written to: {table_path}")


# =========================================================
# Write the table partitioned by city
# =========================================================

joined_table.write \
    .mode("overwrite") \
    .partitionBy("city") \
    .parquet(partitioned_table_path)

print(
    "Partitioned table written to: "
    f"{partitioned_table_path}"
)


# =========================================================
# Stop Spark
# =========================================================

spark.stop()

print("\nWrite process completed successfully.")