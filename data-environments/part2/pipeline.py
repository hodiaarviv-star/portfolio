import os
import subprocess

scripts = [
    "build_directories.py",
    "stages/filter.py",
    "stages/count.py",
    "stages/validate.py",
    "stages/transform.py",
    "stages/query.py"
]

for script in scripts:
    print(f"\nRunning {script}")
    result = subprocess.run(["python3", script])

    if result.returncode != 0:
        raise Exception(f"Script failed: {script}")

print("\nPipeline finished successfully")
