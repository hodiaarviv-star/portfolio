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
# Run one Spark test with one pair of memory fractions
# =========================================================

def run_single_test(memory_fraction, storage_fraction):
    execution_fraction = 1.0 - storage_fraction

    spark = (
        SparkSession.builder
        .appName(
            f"Memory Fractions Test "
            f"{memory_fraction}_{storage_fraction}"
        )
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.driver.memory", "1g")
        .config(
            "spark.memory.fraction",
            str(memory_fraction)
        )
        .config(
            "spark.memory.storageFraction",
            str(storage_fraction)
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    start_time = time.perf_counter()

    df = spark.read.parquet(joined_table_path)

    # =====================================================
    # Query 1
    # =====================================================

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
    print("Five cities with the highest total sales:")
    query1.show(truncate=False)

    # =====================================================
    # Query 2
    # =====================================================

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
    print("Five cities with the highest average order amount:")
    query2.show(truncate=False)

    execution_time = time.perf_counter() - start_time

    print("\n================ SINGLE TEST RESULT ================")
    print(f"spark.memory.fraction: {memory_fraction}")
    print(f"spark.memory.storageFraction: {storage_fraction}")
    print(f"executionFraction: {execution_fraction:.1f}")
    print(f"Execution time: {execution_time:.2f} seconds")

    # Special line used by the parent process
    print(
        f"RESULT|{memory_fraction}|"
        f"{storage_fraction}|"
        f"{execution_fraction:.1f}|"
        f"{execution_time:.2f}"
    )

    spark.stop()


# =========================================================
# Run all nine tests in separate processes
# =========================================================

def run_all_tests():
    memory_fractions = [0.6, 0.7, 0.8]
    storage_fractions = [0.4, 0.5, 0.6]

    results = []

    for memory_fraction in memory_fractions:
        for storage_fraction in storage_fractions:
            print(
                "\n\n=============================================="
            )
            print(
                f"TESTING memory.fraction={memory_fraction}, "
                f"storageFraction={storage_fraction}"
            )
            print(
                "=============================================="
            )

            command = [
                sys.executable,
                os.path.abspath(__file__),
                "--single",
                "--memory-fraction",
                str(memory_fraction),
                "--storage-fraction",
                str(storage_fraction)
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
                    "Test failed for "
                    f"memory.fraction={memory_fraction}, "
                    f"storageFraction={storage_fraction}."
                )
                continue

            result_match = re.search(
                r"RESULT\|([^|]+)\|([^|]+)\|"
                r"([^|]+)\|([0-9.]+)",
                completed_process.stdout
            )

            if result_match:
                tested_memory_fraction = float(
                    result_match.group(1)
                )
                tested_storage_fraction = float(
                    result_match.group(2)
                )
                tested_execution_fraction = float(
                    result_match.group(3)
                )
                tested_time = float(
                    result_match.group(4)
                )

                results.append(
                    {
                        "memory_fraction": tested_memory_fraction,
                        "storage_fraction": tested_storage_fraction,
                        "execution_fraction": tested_execution_fraction,
                        "execution_time": tested_time
                    }
                )
            else:
                print(
                    "Could not read the result for "
                    f"memory.fraction={memory_fraction}, "
                    f"storageFraction={storage_fraction}."
                )

    if not results:
        raise RuntimeError(
            "No memory-fraction test completed successfully."
        )

    # =====================================================
    # Print summary
    # =====================================================

    print(
        "\n\n================ MEMORY FRACTIONS SUMMARY "
        "================"
    )

    for result in results:
        print(
            "memory.fraction="
            f"{result['memory_fraction']:.1f}, "
            "storageFraction="
            f"{result['storage_fraction']:.1f}, "
            "executionFraction="
            f"{result['execution_fraction']:.1f}, "
            "time="
            f"{result['execution_time']:.2f} seconds"
        )

    fastest_result = min(
        results,
        key=lambda item: item["execution_time"]
    )

    print("\nFastest combination:")
    print(
        "spark.memory.fraction: "
        f"{fastest_result['memory_fraction']:.1f}"
    )
    print(
        "spark.memory.storageFraction: "
        f"{fastest_result['storage_fraction']:.1f}"
    )
    print(
        "executionFraction: "
        f"{fastest_result['execution_fraction']:.1f}"
    )
    print(
        "Execution time: "
        f"{fastest_result['execution_time']:.2f} seconds"
    )

    # =====================================================
    # Create scatter plot
    # =====================================================

    x_values = [
        result["memory_fraction"]
        for result in results
    ]

    y_values = [
        result["execution_fraction"]
        for result in results
    ]

    execution_times = [
        result["execution_time"]
        for result in results
    ]

    plt.figure(figsize=(9, 6))

    scatter = plt.scatter(
        x_values,
        y_values,
        s=[
            80 + (time_value * 4)
            for time_value in execution_times
        ],
        c=execution_times
    )

    plt.title(
        "Spark Memory Fractions and Query Execution Time"
    )
    plt.xlabel("spark.memory.fraction")
    plt.ylabel(
        "executionFraction "
        "(1 - spark.memory.storageFraction)"
    )

    plt.xticks([0.6, 0.7, 0.8])
    plt.yticks([0.4, 0.5, 0.6])
    plt.grid(True)

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Execution time (seconds)")

    for result in results:
        plt.annotate(
            f"{result['execution_time']:.2f}s",
            (
                result["memory_fraction"],
                result["execution_fraction"]
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    graph_path = os.path.join(
        results_dir,
        "memory_fractions_scatter.png"
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
    "--memory-fraction",
    type=float
)

parser.add_argument(
    "--storage-fraction",
    type=float
)

args = parser.parse_args()

if args.single:
    if args.memory_fraction is None:
        raise ValueError(
            "Missing --memory-fraction value."
        )

    if args.storage_fraction is None:
        raise ValueError(
            "Missing --storage-fraction value."
        )

    run_single_test(
        args.memory_fraction,
        args.storage_fraction
    )
else:
    run_all_tests()