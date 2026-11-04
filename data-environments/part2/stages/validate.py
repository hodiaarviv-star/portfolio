from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StringType, LongType, IntegerType, ShortType,
    DoubleType, FloatType, DecimalType
)
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = SparkSession.builder.appName("Validate and Clean Data").master("local[*]").getOrCreate()

staged_path = os.path.join(BASE_DIR, "data", "staged")
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")
tables_path = os.path.join(BASE_DIR, "docs", "tables.xlsx")
features_path = os.path.join(BASE_DIR, "docs", "features.xlsx")

os.makedirs(cleaned_path, exist_ok=True)

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


def is_numeric_type(data_type):
    return isinstance(data_type, (LongType, IntegerType, ShortType, DoubleType, FloatType, DecimalType))


def is_string_type(data_type):
    return isinstance(data_type, StringType)


def remove_outliers(df, numeric_columns):
    if not numeric_columns:
        return df

    # One combined action for the quartiles of all numeric columns,
    # instead of a separate collect() per column
    stats = df.select(*numeric_columns).summary("25%", "75%").collect()
    q1_row, q3_row = stats[0], stats[1]

    condition = None
    for column_name in numeric_columns:
        if q1_row[column_name] is None or q3_row[column_name] is None:
            continue

        q1 = float(q1_row[column_name])
        q3 = float(q3_row[column_name])
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        col_condition = (F.col(column_name) >= lower_bound) & (F.col(column_name) <= upper_bound)
        condition = col_condition if condition is None else (condition & col_condition)

    if condition is not None:
        df = df.filter(condition)

    return df


for table_name in tables:
    print(f"\nCleaning table: {table_name}")

    raw_df = spark.read.parquet(f"{staged_path}/{table_name}")
    raw_df = raw_df.persist()

    # 1. Remove duplicates
    df = raw_df.dropDuplicates()

    # Identify columns by type
    string_columns = []
    numeric_columns = []

    for field in df.schema.fields:
        if is_string_type(field.dataType):
            string_columns.append(field.name)
        elif is_numeric_type(field.dataType):
            numeric_columns.append(field.name)

    # 2. Missing values
    # Qualitative/string columns: remove rows with missing values
    for column_name in string_columns:
        df = df.filter(F.col(column_name).isNotNull())

    # Re-persist: this is the version repeatedly scanned by the actions
    # below (mean fill, outlier quartiles, min/max casts). Kept as a
    # separate reference (persisted_df) so the unpersist() at the end of
    # the loop releases the correct cached DataFrame even after `df`
    # itself gets reassigned several more times below.
    raw_df.unpersist()
    df = df.persist()
    persisted_df = df

    # Quantitative/numeric columns: fill missing values with average
    # One combined aggregation for all numeric columns, instead of a
    # separate collect() per column
    if numeric_columns:
        mean_row = df.select([F.mean(F.col(c)).alias(c) for c in numeric_columns]).first()
        fill_values = {c: mean_row[c] for c in numeric_columns if mean_row[c] is not None}

        if fill_values:
            df = df.fillna(fill_values)

    # 3. Outliers - remove rows with outliers in numeric columns
    df = remove_outliers(df, numeric_columns)

    # 4. Reduce numeric variable types to the smallest type that fits the actual data range
    # One combined min/max aggregation for all Long/Integer columns,
    # instead of a separate collect() per column
    long_int_columns = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, (LongType, IntegerType))
    ]

    minmax_row = None
    if long_int_columns:
        minmax_exprs = []
        for column_name in long_int_columns:
            minmax_exprs.append(F.min(F.col(column_name)).alias(f"min_{column_name}"))
            minmax_exprs.append(F.max(F.col(column_name)).alias(f"max_{column_name}"))
        minmax_row = df.select(minmax_exprs).first()

    for field in df.schema.fields:
        column_name = field.name
        data_type = field.dataType

        if isinstance(data_type, (LongType, IntegerType)):
            col_min = minmax_row[f"min_{column_name}"]
            col_max = minmax_row[f"max_{column_name}"]

            if col_min is None or col_max is None:
                continue

            if -32768 <= col_min and col_max <= 32767:
                df = df.withColumn(column_name, F.col(column_name).cast(ShortType()))
            elif isinstance(data_type, LongType) and -2147483648 <= col_min and col_max <= 2147483647:
                df = df.withColumn(column_name, F.col(column_name).cast(IntegerType()))
        elif isinstance(data_type, (DoubleType, DecimalType)):
            df = df.withColumn(column_name, F.col(column_name).cast(FloatType()))

    # Save cleaned table
    df.write.mode("overwrite").option("compression", "snappy").parquet(f"{cleaned_path}/{table_name}")

    print(f"Finished cleaning: {table_name}, rows: {df.count()}, columns: {len(df.columns)}")

    persisted_df.unpersist()


# ===========================
# Update tables.xlsx
# ===========================
tables_df = pd.read_excel(tables_path)

for table_name in tables:
    cleaned_df = spark.read.parquet(f"{cleaned_path}/{table_name}")

    cleaned_records = cleaned_df.count()
    cleaned_columns = len(cleaned_df.columns)

    mask = tables_df["table"] == table_name

    tables_df.loc[mask, "cleaned_records"] = cleaned_records
    tables_df.loc[mask, "cleaned_columns"] = cleaned_columns

    staged_records = tables_df.loc[mask, "staged_records"].values[0]
    staged_columns = tables_df.loc[mask, "staged_columns"].values[0]

    cleaned_staged = int(
        100 * (cleaned_records * cleaned_columns) /
        (staged_records * staged_columns)
    )

    tables_df.loc[mask, "cleaned/staged"] = cleaned_staged

tables_df.to_excel(tables_path, index=False)


# ===========================
# Update features.xlsx data types after cleaning
# ===========================
features_df = pd.read_excel(features_path)

for table_name in tables:
    cleaned_df = spark.read.parquet(f"{cleaned_path}/{table_name}")

    for field in cleaned_df.schema.fields:
        mask = (
            (features_df["table"] == table_name) &
            (features_df["feature"] == field.name)
        )
        features_df.loc[mask, "data_type"] = str(field.dataType)

features_df.to_excel(features_path, index=False)

spark.stop()
