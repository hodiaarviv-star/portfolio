import os, time, getpass, sqlite3
import pandas as pd
import findspark

findspark.init()

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

base_dir = f"/home/{getpass.getuser()}/task1"
data_dir = os.path.join(base_dir, "data")
results_dir = os.path.join(base_dir, "results")

spark = (
    SparkSession.builder
    .appName("ex1")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

sfs = [0.5, 1, 2]

# פורמטים לפי מוסכמת השם ב-generateTPCH.py: {compression}_{table}.{format}
formats = [
    {"compression": "uncompressed", "format": "parquet"},
    {"compression": "uncompressed", "format": "csv"},
    {"compression": "snappy", "format": "parquet"},
]

def read_table(sf_dir, table, fmt):
    file_name = f"{fmt['compression']}_{table}.{fmt['format']}"
    file_path = os.path.join(sf_dir, file_name)

    if fmt["format"] == "csv":
        df = spark.read.option("header", True).option("inferSchema", True).csv(file_path)
    else:
        df = spark.read.parquet(file_path)

    return df, file_name

results = []

for sf in sfs:

    print(f"\nRunning queries for scale factor {sf}")

    sf_dir = os.path.join(data_dir, f"sf{sf}")

    for fmt in formats:

        print(f"format: {fmt['compression']} {fmt['format']}")

        lineitem_df, lineitem_file = read_table(sf_dir, "lineitem", fmt)
        # lineitem_df משמש בשלוש פעולות (groupOrder, groupOrderJoin, countDistinct) -
        # persist כדי למנוע קריאה חוזרת של הקובץ מהדיסק שלוש פעמים
        lineitem_df.persist()

        # 1. groupOrder
        start = time.time()

        calculation = (
            lineitem_df
            .groupBy("l_partkey")
            .agg(F.mean("l_quantity").alias("avg_quantity"))
            .orderBy(F.desc("avg_quantity"))
            .limit(1)
        )

        result = calculation.collect()[0][0]
        duration = round(time.time() - start, 2)

        print(f"groupOrder result: {result}")
        print("groupOrder duration:", duration)

        results.append({
            "name": lineitem_file,
            "scale": float(sf),
            "action": "groupOrder",
            "duration": float(duration)
        })

        # קריאת part - באותו פורמט כמו lineitem, כדי שה-join יחבר קבצים מאותו פורמט
        part_df, part_file = read_table(sf_dir, "part", fmt)

        # 2. groupOrderJoin
        start = time.time()

        calculation = (
            part_df
            .join(lineitem_df, part_df.p_partkey == lineitem_df.l_partkey, "inner")
            .groupBy(part_df.p_name)
            .agg(F.mean(lineitem_df.l_quantity).alias("avg_quantity"))
            .orderBy(F.desc("avg_quantity"))
            .limit(1)
        )

        result = calculation.collect()[0][0]
        duration = round(time.time() - start, 2)

        print(f"groupOrderJoin result: {result}")
        print("groupOrderJoin duration:", duration)

        results.append({
            "name": part_file,
            "scale": float(sf),
            "action": "groupOrderJoin",
            "duration": float(duration)
        })

        # 3. countDistinct
        start = time.time()

        calculation = lineitem_df.agg(
            F.countDistinct(lineitem_df.l_partkey).alias("distinct_parts")
        )

        result = calculation.collect()[0][0]
        duration = round(time.time() - start, 2)

        print(f"countDistinct result: {result}")
        print("countDistinct duration:", duration)

        results.append({
            "name": lineitem_file,
            "scale": float(sf),
            "action": "countDistinct",
            "duration": float(duration)
        })

        lineitem_df.unpersist()

# DataFrame
df = pd.DataFrame(results)

print(df)

conn = sqlite3.connect(os.path.join(results_dir, "monitor.db"))
cursor = conn.cursor()

# הטבלה כבר נוצרה ע"י db_tables.py; מוחקים כאן רק את תוצאות השאילתות
# מריצות קודמות (לא את שורות ה-write שנכתבו ע"י generateTPCH.py)
cursor.execute("DELETE FROM calculations WHERE action != 'write'")

for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO calculations
        (name, scale, action, duration)
        VALUES (?, ?, ?, ?)
        """,
        (
            row["name"],
            float(row["scale"]),
            row["action"],
            float(row["duration"])
        )
    )

conn.commit()

calculations_df = pd.read_sql_query(
    "SELECT * FROM calculations",
    conn
)

print(calculations_df)

calculations_df.to_excel(
    os.path.join(results_dir, "calculations.xlsx"),
    index=False
)

conn.close()
spark.stop()

print("DONE")