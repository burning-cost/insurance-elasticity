# Databricks notebook source
# Runs the full test suite for insurance-elasticity.
# Dependencies installed inline via %pip.

# COMMAND ----------

# MAGIC %pip install "insurance-elasticity[all]==0.1.0" pytest --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys, os

# Clone the repo to get the tests (PyPI package doesn't include tests/)
os.makedirs("/tmp/ie_repo", exist_ok=True)
result = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burning-cost/insurance-elasticity.git",
     "/tmp/ie_repo"],
    capture_output=True, text=True
)
print("Clone stdout:", result.stdout)
print("Clone stderr:", result.stderr)
print("Clone returncode:", result.returncode)

# COMMAND ----------

# Run the tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
output = result.stdout
if len(output) > 10000:
    output = output[-10000:]
print(output)
if result.stderr:
    stderr = result.stderr[-2000:]
    print("STDERR:", stderr)
print("Return code:", result.returncode)

# COMMAND ----------

# Summary: pass exit code to final cell result
if result.returncode == 0:
    print("ALL TESTS PASSED")
else:
    raise Exception(f"Tests FAILED with return code {result.returncode}")
