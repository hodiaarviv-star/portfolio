from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = SparkSession.builder.appName("Filter Data").master("local[*]").getOrCreate()

# Paths
raw_path = os.path.join(BASE_DIR, "data", "raw")
staged_path = os.path.join(BASE_DIR, "data", "staged")
docs_path = os.path.join(BASE_DIR, "docs", "tables.xlsx")

os.makedirs(staged_path, exist_ok=True)

tables_info = []


def save_filtered_table(table_name, raw_df, filtered_df):
    raw_records = raw_df.count()
    raw_columns = len(raw_df.columns)

    filtered_df = filtered_df.persist()

    filtered_records = filtered_df.count()
    filtered_columns = len(filtered_df.columns)

    percent_reduction = int(
        100 * (filtered_records * filtered_columns) / (raw_records * raw_columns)
    )

    filtered_df.write.mode("overwrite").option("compression", "snappy").parquet(f"{staged_path}/{table_name}")

    filtered_df.unpersist()

    tables_info.append({
        "table": table_name,
        "raw_records": raw_records,
        "raw_columns": raw_columns,
        "staged_records": filtered_records,
        "staged_columns": filtered_columns,
        "staged/raw": percent_reduction,
        "cleaned_records": "",
        "cleaned_columns": "",
        "cleaned/staged": "",
        "transformed_records": "",
        "transformed_columns": "",
        "transformed/cleaned": ""
    })


# ======================
# ZOMATO
# ======================

orders = spark.read.parquet(f"{raw_path}/zomato/orders.parquet")
orders_filtered = orders.filter(F.col("sales_amount") > 0).select(
    "order_date", "sales_qty", "sales_amount", "currency", "user_id", "r_id"
)
save_filtered_table("zomato_orders", orders, orders_filtered)

restaurant = spark.read.parquet(f"{raw_path}/zomato/restaurant.parquet")
restaurant_filtered = restaurant.select(
    "id", "name", "city", "rating", "rating_count", "cost", "cuisine"
).filter(F.col("id").isNotNull())
save_filtered_table("zomato_restaurant", restaurant, restaurant_filtered)

users = spark.read.parquet(f"{raw_path}/zomato/users.parquet")
users_filtered = users.select(
    "user_id", "Age", "Gender", "Marital Status", "Occupation",
    "Monthly Income", "Educational Qualifications", "Family size"
).filter(F.col("user_id").isNotNull())
save_filtered_table("zomato_users", users, users_filtered)

menu = spark.read.parquet(f"{raw_path}/zomato/menu.parquet")
menu_filtered = menu.select(
    "menu_id", "r_id", "f_id", "cuisine", "price"
).filter(F.col("price").isNotNull())
save_filtered_table("zomato_menu", menu, menu_filtered)

food = spark.read.parquet(f"{raw_path}/zomato/food.parquet")
food_filtered = food.select(
    "f_id", "item", "veg_or_non_veg"
).filter(F.col("f_id").isNotNull())
save_filtered_table("zomato_food", food, food_filtered)


# ======================
# NBA
# ======================

team_box = spark.read.parquet(f"{raw_path}/NBA/datasets/2012-18_teamBoxScore.parquet")
team_box_filtered = team_box.select(
    "gmDate", "gmTime", "seasTyp", "teamAbbr", "teamConf", "teamDiv",
    "teamLoc", "teamRslt", "teamPTS", "teamAST", "teamTO",
    "teamTRB", "teamFG%", "team3P%", "opptAbbr", "opptPTS",
    "poss", "pace"
)
save_filtered_table("nba_teamBoxScore", team_box, team_box_filtered)

player_box = spark.read.parquet(f"{raw_path}/NBA/datasets/2012-18_playerBoxScore.parquet")
player_box_filtered = player_box.select(
    "gmDate", "gmTime", "seasTyp", "playDispNm", "teamAbbr",
    "teamLoc", "teamRslt", "playStat", "playMin", "playPos",
    "playPTS", "playAST", "playTRB", "playFG%", "opptAbbr"
).filter(F.col("playMin") > 0)
save_filtered_table("nba_playerBoxScore", player_box, player_box_filtered)

standings = spark.read.parquet(f"{raw_path}/NBA/datasets/2012-18_standings.parquet")
standings_filtered = standings.select(
    "stDate", "teamAbbr", "rank", "gameWon", "gameLost",
    "stk", "stkType", "stkTot", "homeWin", "homeLoss",
    "awayWin", "awayLoss", "ptsScore", "ptsAllow", "ptsDiff"
)
save_filtered_table("nba_standings", standings, standings_filtered)


# Save documentation to tables.xlsx
tables_df = pd.DataFrame(tables_info)
print(tables_df)
tables_df.to_excel(docs_path, index=False)

spark.stop()
