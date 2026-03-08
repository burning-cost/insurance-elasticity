# Databricks notebook source
# Runs the full test suite for insurance-elasticity.
# Install from PyPI (after publish) or from GitHub.

# COMMAND ----------

# MAGIC %pip install "insurance-elasticity[all]" pytest --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys, os

# Clone the repo to get the tests
os.makedirs("/tmp/ie_tests", exist_ok=True)
result = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burning-cost/insurance-elasticity.git",
     "/tmp/ie_tests"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# COMMAND ----------

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
    capture_output=True, text=True,
    cwd="/tmp/ie_tests",
)
output = result.stdout
if len(output) > 10000:
    output = output[-10000:]
print(output)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
