# Databricks notebook source
# This notebook installs the library from GitHub and runs the test suite.

# COMMAND ----------

# MAGIC %pip install "insurance-elasticity[all] @ git+https://github.com/burning-cost/insurance-elasticity.git" pytest --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "--tb=short", "-v",
     "--pyargs", "tests"],
    capture_output=True, text=True,
    cwd="/tmp",
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
