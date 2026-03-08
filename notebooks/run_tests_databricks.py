# Databricks notebook source
# Runs the full test suite for insurance-elasticity.

# COMMAND ----------

# MAGIC %pip install "econml>=0.15" "catboost>=1.2" "insurance-elasticity==0.1.0" "pytest"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys, os

os.makedirs("/tmp/ie_repo", exist_ok=True)
clone_result = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burning-cost/insurance-elasticity.git",
     "/tmp/ie_repo"],
    capture_output=True, text=True
)
print("Clone:", clone_result.returncode, clone_result.stderr[:500])

# COMMAND ----------

test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=long", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
full_out = test_result.stdout + "\n" + test_result.stderr
print(full_out[-8000:])

# COMMAND ----------

# Pass summary out
lines = [l for l in full_out.split("\n") if "passed" in l or "failed" in l or "error" in l.lower()]
summary = "\n".join(lines[-20:])
status = "PASSED" if test_result.returncode == 0 else "FAILED"
dbutils.notebook.exit(f"{status} rc={test_result.returncode}\n{summary}\n\nFULL OUTPUT (last 3000):\n{full_out[-3000:]}")
