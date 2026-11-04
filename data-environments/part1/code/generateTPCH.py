import os, time, getpass, duckdb, shutil, sqlite3
import pandas as pd
import findspark

findspark.init()

from pyspark.sql import SparkSession

home_dir = f'/home/{getpass.getuser()}'
task_dir = os.path.join(home_dir, 'task1')
data_dir = os.path.join(task_dir, 'data')
results_dir = os.path.join(task_dir, 'results')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

conn = sqlite3.connect(os.path.join(results_dir, 'monitor.db'))
cursor = conn.cursor()

cursor.execute("DELETE FROM files")
cursor.execute("DELETE FROM calculations WHERE action = 'write'")
conn.commit()

con = duckdb.connect(database=':memory:')
spark = (
    SparkSession.builder
    .appName("ex1")
    .master('local[*]')
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

tables = [
    "customer", "lineitem", "nation", "orders",
    "part", "partsupp", "region", "supplier"
]

sfs = [0.5, 1, 2]

def insert_file(name, file_format, sf, compression, rows, columns, duration):
    cursor.execute(
        """
        INSERT INTO files
        (name, format, scale, compression, rows, columns, write_duration)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (name, file_format, float(sf), compression, rows, columns, float(duration))
    )
    conn.commit()

def insert_calculation(name, sf, action, duration):
    cursor.execute(
        """
        INSERT INTO calculations
        (name, scale, action, duration)
        VALUES (?, ?, ?, ?)
        """,
        (name, float(sf), action, float(duration))
    )
    conn.commit()

def write_df(df, file_path, file_format, compression=None):
    print(f"writing to {file_path}")

    start = time.time()

    if file_format == 'parquet':
        df.write.mode("overwrite").option("compression", compression).parquet(file_path)

    elif file_format == 'csv':
        df.write.mode("overwrite").option("header", True).csv(file_path)

    duration = round(time.time() - start, 2)

    print(f'write duration: {duration}')
    os.system(f'du -sh {file_path}')

    return duration

for sf in sfs:

    print(f'\nData for scale factor {sf}')

    sf_dir = os.path.join(data_dir, f'sf{sf}')

    if os.path.exists(sf_dir):
        shutil.rmtree(sf_dir)

    os.makedirs(sf_dir, exist_ok=True)

    print('tpch generation')
    start = time.time()

    con.execute(f"CALL dbgen(sf={sf})")

    print(f'generation time = {round(time.time() - start, 2)}')
    print(con.execute("show tables").fetchall())
    print('duckdb tables generated')
    print('copy duckdb tables to parquet and CSV files')

    for table in tables:

        print(f'read {table}')

        a_duck = con.query(f"SELECT * FROM {table}")
        pandas_df = a_duck.to_df()

        tmp = os.path.join(task_dir, 'tmp.parquet')
        pandas_df.to_parquet(tmp)

        df_spark = spark.read.parquet(tmp)
        # df_spark משמש ב-count() ובשלוש כתיבות נפרדות -
        # persist כדי למנוע קריאה חוזרת של tmp.parquet ארבע פעמים
        df_spark.persist()

        rows = int(df_spark.count())
        columns = int(len(df_spark.columns))

        print(f'Table Shape: {rows} rows, {columns} columns')

        # 1. parquet uncompressed
        file_format = 'parquet'
        compression = 'uncompressed'
        file = f"{compression}_{table}.{file_format}"
        file_path = os.path.join(sf_dir, file)

        duration = write_df(df_spark, file_path, file_format, compression)

        insert_file(file, file_format, sf, compression, rows, columns, duration)
        insert_calculation(file, sf, 'write', duration)

        # 2. csv uncompressed
        file_format = 'csv'
        compression = 'uncompressed'
        file = f"{compression}_{table}.{file_format}"
        file_path = os.path.join(sf_dir, file)

        duration = write_df(df_spark, file_path, file_format, compression)

        insert_file(file, file_format, sf, compression, rows, columns, duration)
        insert_calculation(file, sf, 'write', duration)

        # 3. parquet snappy compressed
        file_format = 'parquet'
        compression = 'snappy'
        file = f"{compression}_{table}.{file_format}"
        file_path = os.path.join(sf_dir, file)

        duration = write_df(df_spark, file_path, file_format, compression)

        insert_file(file, file_format, sf, compression, rows, columns, duration)
        insert_calculation(file, sf, 'write', duration)

        df_spark.unpersist()

        del df_spark
        del pandas_df

        if os.path.exists(tmp):
            os.remove(tmp)

    print('duckdb tables dropped')

    for table in tables:
        con.execute(f"DROP TABLE IF EXISTS {table}")

conn.commit()

files = pd.read_sql_query("SELECT * FROM files", conn)

print(files)

files.to_excel(
    os.path.join(results_dir, 'files.xlsx'),
    index=False
)

conn.close()
con.close()
spark.stop()

print("DONE")