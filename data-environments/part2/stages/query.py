from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = SparkSession.builder.appName("Business Queries").master("local[*]").getOrCreate()

transformed_path = os.path.join(BASE_DIR, "data", "transformed")
results_path = os.path.join(BASE_DIR, "results")

os.makedirs(results_path, exist_ok=True)


def save_query_result(df, query_id):
    output_path = f"{results_path}/query{query_id}.xlsx"
    df.toPandas().to_excel(output_path, index=False)
    print(f"Saved query{query_id}: {output_path}")


# =========================
# Read transformed tables
# =========================
orders = spark.read.parquet(f"{transformed_path}/zomato_orders")
restaurant = spark.read.parquet(f"{transformed_path}/zomato_restaurant")
users = spark.read.parquet(f"{transformed_path}/zomato_users")
menu = spark.read.parquet(f"{transformed_path}/zomato_menu")
food = spark.read.parquet(f"{transformed_path}/zomato_food")

team_box = spark.read.parquet(f"{transformed_path}/nba_teamBoxScore")
player_box = spark.read.parquet(f"{transformed_path}/nba_playerBoxScore")
standings = spark.read.parquet(f"{transformed_path}/nba_standings")

orders = orders.withColumn("r_id_long", F.col("r_id").cast("long"))

# Persist tables that are read once from disk but reused across several queries below
orders = orders.persist()
restaurant = restaurant.persist()
team_box = team_box.persist()
standings = standings.persist()


# Query 1 - single fact/dim join, GroupBy
query1 = orders.join(restaurant, orders["r_id_long"] == restaurant["id"], "inner") \
    .groupBy("city") \
    .agg(
        F.sum("sales_amount").alias("total_sales"),
        F.count("*").alias("number_of_orders"),
        F.round(F.avg("avg_item_price"), 2).alias("avg_item_price")
    ) \
    .orderBy(F.desc("total_sales")) \
    .limit(5)

save_query_result(query1, 1)


# Query 2 - single table, window function (no join, no GroupBy)
city_window = Window.partitionBy("city").orderBy(F.desc("rating"))

query2 = restaurant.withColumn("rank_in_city", F.dense_rank().over(city_window)) \
    .filter(F.col("rank_in_city") == 1) \
    .select("city", "name", "rating", "rating_count") \
    .orderBy(F.desc("rating")) \
    .limit(5)

save_query_result(query2, 2)


# Query 3 - fact table (orders) joined with 2 dimension tables (users, restaurant), GroupBy
query3 = orders.join(users, "user_id", "inner") \
    .join(restaurant, orders["r_id_long"] == restaurant["id"], "inner") \
    .groupBy("Occupation", restaurant["city"]) \
    .agg(
        F.round(F.avg("sales_amount"), 2).alias("avg_order_amount"),
        F.count("*").alias("number_of_orders")
    ) \
    .filter(F.col("number_of_orders") >= 20) \
    .orderBy(F.desc("avg_order_amount")) \
    .limit(5)

save_query_result(query3, 3)


# Query 4 - single 2-table join (menu+food), no GroupBy
query4 = menu.join(food, "f_id", "inner") \
    .select(
        food["item"].alias("food_item"),
        food["veg_or_non_veg"],
        menu["cuisine"],
        menu["price"]
    ) \
    .orderBy(F.desc("price")) \
    .limit(5)

save_query_result(query4, 4)


# Query 5 - single table, GroupBy, uses derived column point_diff
query5 = team_box.groupBy("teamAbbr") \
    .agg(
        F.round(F.avg("teamPTS"), 2).alias("avg_points_per_game"),
        F.round(F.avg("point_diff"), 2).alias("avg_point_diff"),
        F.count("*").alias("games_count")
    ) \
    .orderBy(F.desc("avg_points_per_game")) \
    .limit(5)

save_query_result(query5, 5)


# Query 6 - join (player_box + team_box), no GroupBy, uses derived column point_diff
query6 = player_box.join(
        team_box,
        (player_box["teamAbbr"] == team_box["teamAbbr"]) & (player_box["gmDate"] == team_box["gmDate"]),
        "inner"
    ) \
    .select(
        player_box["gmDate"],
        "playDispNm",
        player_box["teamAbbr"],
        "playPTS",
        team_box["point_diff"]
    ) \
    .orderBy(F.desc("playPTS")) \
    .limit(5)

save_query_result(query6, 6)


# Query 7 - join (team_box + standings), no GroupBy, uses derived columns is_home_game and win_rate
query7 = team_box.filter(F.col("is_home_game") == 1) \
    .join(
        standings,
        (team_box["teamAbbr"] == standings["teamAbbr"]) & (team_box["gmDate"] == standings["stDate"]),
        "inner"
    ) \
    .select(
        team_box["gmDate"],
        team_box["teamAbbr"],
        "teamPTS",
        team_box["point_diff"],
        standings["win_rate"]
    ) \
    .orderBy(F.desc("point_diff")) \
    .limit(5)

save_query_result(query7, 7)


# Query 8 - single table, window function, uses derived column win_rate
window_spec = Window.orderBy(F.desc("stkTot"))

query8 = standings.filter(F.col("stkType") == "win") \
    .withColumn("streak_rank", F.dense_rank().over(window_spec)) \
    .select("stDate", "teamAbbr", "stk", "stkType", "stkTot", "win_rate", "streak_rank") \
    .orderBy(F.asc("streak_rank"), F.desc("stkTot")) \
    .limit(5)

save_query_result(query8, 8)


orders.unpersist()
restaurant.unpersist()
team_box.unpersist()
standings.unpersist()

spark.stop()