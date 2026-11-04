from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, LongType, IntegerType, ShortType, DoubleType, FloatType, DecimalType
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spark = SparkSession.builder.appName("Count Data Quality Metrics").master("local[*]").getOrCreate()

staged_path = os.path.join(BASE_DIR, "data", "staged")
features_path = os.path.join(BASE_DIR, "docs", "features.xlsx")

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

results = []


def calc_outlier_bounds(df, numeric_columns):
    """One combined quartile action for all numeric columns, instead of one per column."""
    if not numeric_columns:
        return {}

    stats = df.select(*numeric_columns).summary("25%", "75%").collect()
    q1_row, q3_row = stats[0], stats[1]

    bounds = {}
    for column_name in numeric_columns:
        if q1_row[column_name] is None or q3_row[column_name] is None:
            continue

        q1 = float(q1_row[column_name])
        q3 = float(q3_row[column_name])
        iqr = q3 - q1

        bounds[column_name] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return bounds


for table_name in tables:
    df = spark.read.parquet(f"{staged_path}/{table_name}")
    df = df.persist()

    total_records = df.count()

    duplicate_count = total_records - df.dropDuplicates().count()
    print(f"{table_name}: {duplicate_count} duplicate records")

    string_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, StringType)]
    numeric_columns = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, (LongType, IntegerType, ShortType, DoubleType, FloatType, DecimalType))
    ]

    # One combined action for null counts of ALL columns, instead of one per column
    null_row = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).first()

    # One combined action for distinct counts of all string columns, instead of one per column
    distinct_row = None
    if string_columns:
        distinct_row = df.select(
            [F.countDistinct(F.col(c)).alias(c) for c in string_columns]
        ).first()

    # One combined quartile action for all numeric columns
    bounds = calc_outlier_bounds(df, numeric_columns)

    # One combined action for outlier counts of all numeric columns, instead of
    # a separate filter().count() per column
    outlier_row = None
    if bounds:
        outlier_row = df.select(
            [
                F.sum(
                    F.when((F.col(c) < lb) | (F.col(c) > ub), 1).otherwise(0)
                ).alias(c)
                for c, (lb, ub) in bounds.items()
            ]
        ).first()

    for field in df.schema.fields:
        column_name = field.name
        data_type = field.dataType

        null_count = null_row[column_name] or 0
        nulls_ratio = int(100 * null_count / total_records) if total_records > 0 else 0

        outliers_ratio = ""
        distinct_count = ""

        if isinstance(data_type, StringType):
            distinct_count = distinct_row[column_name] if distinct_row is not None else 0

        elif isinstance(data_type, (LongType, IntegerType, ShortType, DoubleType, FloatType, DecimalType)):
            if total_records == 0:
                outliers_ratio = 0
            elif column_name in bounds and outlier_row is not None:
                outliers_count = outlier_row[column_name] or 0
                outliers_ratio = int(100 * outliers_count / total_records)
            else:
                outliers_ratio = 0

        results.append({
            "feature": column_name,
            "table": table_name,
            "data_type": str(data_type),
            "nulls_ratio": nulls_ratio,
            "outliers_ratio": outliers_ratio,
            "distinct_count": distinct_count
        })

    df.unpersist()


features_df = pd.DataFrame(results)

print(features_df)

features_df.to_excel(features_path, index=False)

spark.stop()
