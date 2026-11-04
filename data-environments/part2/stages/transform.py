from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = SparkSession.builder.appName("Transform Data").master("local[*]").getOrCreate()

cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")
transformed_path = os.path.join(BASE_DIR, "data", "transformed")
tables_path = os.path.join(BASE_DIR, "docs", "tables.xlsx")
os.makedirs(transformed_path, exist_ok=True)

tables = [
    "zomato_orders",
    "zomato_restaurant",
    "zomato_users",
    "zomato_menu",
    "zomato_food",
    "nba_teamBoxScore",
    "nba_playerBoxScore",
    "nba_standings"
]


def save_transformed_table(table_name, df):
    df.write.mode("overwrite").option("compression", "snappy").parquet(f"{transformed_path}/{table_name}")
    print(f"Saved transformed table: {table_name}, rows: {df.count()}, columns: {len(df.columns)}")


for table_name in tables:
    print(f"\nTransforming table: {table_name}")

    df = spark.read.parquet(f"{cleaned_path}/{table_name}")

    if table_name == "zomato_orders":
        df = df.withColumn(
            "avg_item_price",
            F.round(F.col("sales_amount") / F.col("sales_qty"), 2)
        )

        # אחרי שיצרנו מחיר ממוצע לפריט, לא חייבים לשמור sales_qty לשאילתות שלנו
        df = df.drop("sales_qty")

    elif table_name == "nba_teamBoxScore":
        df = df.withColumn(
            "point_diff",
            F.col("teamPTS") - F.col("opptPTS")
        ).withColumn(
            "is_home_game",
            F.when(F.col("teamLoc") == "Home", 1).otherwise(0)
        )

        # הורדת שדות שלא נדרשים ישירות אחרי יצירת point_diff / is_home_game
        df = df.drop("opptPTS", "teamLoc")

    elif table_name == "nba_standings":
        df = df.withColumn(
            "win_rate",
            F.round(F.col("gameWon") / (F.col("gameWon") + F.col("gameLost")), 3)
        )

        # הורדת שדות שלא נדרשים ישירות אחרי יצירת win_rate
        df = df.drop("gameWon", "gameLost")

    df = df.persist()
    save_transformed_table(table_name, df)
    df.unpersist()


# ===========================
# Update tables.xlsx
# ===========================

tables_df = pd.read_excel(tables_path)

for table_name in tables:
    transformed_df = spark.read.parquet(f"{transformed_path}/{table_name}")

    transformed_records = transformed_df.count()
    transformed_columns = len(transformed_df.columns)

    mask = tables_df["table"] == table_name

    tables_df.loc[mask, "transformed_records"] = transformed_records
    tables_df.loc[mask, "transformed_columns"] = transformed_columns

    cleaned_records = tables_df.loc[mask, "cleaned_records"].values[0]
    cleaned_columns = tables_df.loc[mask, "cleaned_columns"].values[0]

    transformed_cleaned = int(
        100 * (transformed_records * transformed_columns) /
        (cleaned_records * cleaned_columns)
    )

    tables_df.loc[mask, "transformed/cleaned"] = transformed_cleaned

tables_df.to_excel(tables_path, index=False)

spark.stop()