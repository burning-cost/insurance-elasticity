# Databricks notebook source
# Runs the full test suite for insurance-elasticity.

# COMMAND ----------

# MAGIC %pip install "econml>=0.15" "catboost>=1.2" "insurance-elasticity==0.1.0" "pytest" --quiet

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
if clone_result.returncode != 0:
    dbutils.notebook.exit(f"CLONE FAILED: {clone_result.stderr}")

print("Clone OK.")

# COMMAND ----------

test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
out = test_result.stdout + test_result.stderr
# Keep last 5000 chars for notebook output cell
if len(out) > 5000:
    out = out[-5000:]
print(out)

# COMMAND ----------

# Exit with test output so get_run_output can retrieve it
status = "PASSED" if test_result.returncode == 0 else "FAILED"
exit_msg = f"Tests {status} (rc={test_result.returncode})\n\n{out}"
dbutils.notebook.exit(exit_msg[:4000])
