import argparse
import os
import re
import subprocess
import sys
import time

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, desc
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import sum as spark_sum


# =========================================================
# Relative project paths
# =========================================================

root = os.getcwd()

data_dir = os.path.join(root, "data")
results_dir = os.path.join(root, "results")

joined_table_path = os.path.join(data_dir, "table.parquet")

os.makedirs(results_dir, exist_ok=True)


# =========================================================
# Run one Spark test with one driver-memory value
# =========================================================

def run_single_test(memory_value):
    spark = (
        SparkSession.builder
        .appName(f"JVM Memory Test {memory_value}")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.driver.memory", memory_value)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    start_time = time.perf_counter()

    df = spark.read.parquet(joined_table_path)

    query1 = (
        df.groupBy("city")
        .agg(
            spark_sum("sales_amount").alias("total_sales"),
            count("*").alias("number_of_orders")
        )
        .orderBy(desc("total_sales"))
        .limit(5)
    )

    print("\n================ QUERY 1 ================")
    query1.show(truncate=False)

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

    print("\n================ QUERY 2 ================")
    query2.show(truncate=False)

    execution_time = time.perf_counter() - start_time

    print("\n================ SINGLE TEST RESULT ================")
    print(f"Driver memory: {memory_value}")
    print(f"Execution time: {execution_time:.2f} seconds")

    # This special line is read by the parent process
    print(f"RESULT|{memory_value}|{execution_time:.2f}")

    spark.stop()


# =========================================================
# Run all four tests in separate Python/Spark processes
# =========================================================

def run_all_tests():
    memory_values = ["500m", "1g", "1500m", "2g"]

    memory_in_gb = {
        "500m": 0.5,
        "1g": 1.0,
        "1500m": 1.5,
        "2g": 2.0
    }

    execution_times = {}

    for memory_value in memory_values:
        print(
            f"\n\n================ TESTING {memory_value} ================"
        )

        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--single",
            "--memory",
            memory_value
        ]

        completed_process = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True
        )

        print(completed_process.stdout)

        if completed_process.stderr:
            print(completed_process.stderr)

        if completed_process.returncode != 0:
            print(
                f"Test failed for driver memory {memory_value}. "
                "The remaining tests will continue."
            )
            continue

        result_match = re.search(
            r"RESULT\|([^|]+)\|([0-9.]+)",
            completed_process.stdout
        )

        if result_match:
            tested_memory = result_match.group(1)
            tested_time = float(result_match.group(2))
            execution_times[tested_memory] = tested_time
        else:
            print(
                f"Could not find the execution time for {memory_value}."
            )

    if not execution_times:
        raise RuntimeError("No JVM memory test completed successfully.")

    print("\n\n================ JVM MEMORY SUMMARY ================")

    for memory_value in memory_values:
        if memory_value in execution_times:
            print(
                f"{memory_value}: "
                f"{execution_times[memory_value]:.2f} seconds"
            )

    fastest_memory = min(
        execution_times,
        key=execution_times.get
    )

    print(f"\nFastest driver memory: {fastest_memory}")
    print(
        f"Fastest execution time: "
        f"{execution_times[fastest_memory]:.2f} seconds"
    )

    # =====================================================
    # Create scatter plot
    # =====================================================

    available_values = [
        value
        for value in memory_values
        if value in execution_times
    ]

    x_values = [
        memory_in_gb[value]
        for value in available_values
    ]

    y_values = [
        execution_times[value]
        for value in available_values
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_values, y_values, s=80)

    plt.title("JVM Driver Memory and Query Execution Time")
    plt.xlabel("spark.driver.memory (GB)")
    plt.ylabel("Execution time (seconds)")
    plt.xticks(x_values)
    plt.grid(True)

    graph_path = os.path.join(
        results_dir,
        "jvm_memory_scatter.png"
    )

    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"\nGraph saved to: {graph_path}")


# =========================================================
# Command-line arguments
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--single",
    action="store_true"
)

parser.add_argument(
    "--memory",
    type=str
)

args = parser.parse_args()

if args.single:
    if not args.memory:
        raise ValueError(
            "A memory value must be supplied with --memory."
        )

    run_single_test(args.memory)
else:
    run_all_tests()