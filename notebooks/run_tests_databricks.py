# Databricks notebook source
# Runs the full test suite for insurance-elasticity.
# Uses subprocess pip (not %pip magic) to avoid Databricks serverless restrictions.

# COMMAND ----------

import subprocess, sys, os

def pip_install(*packages):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + list(packages),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-4000:])
        print("STDERR:", result.stderr[-4000:])
        raise RuntimeError(f"pip install failed for: {packages}")
    print(f"OK: {packages}")
    return result.stdout

print("Python:", sys.version)

pip_install("econml>=0.15")
pip_install("catboost>=1.2")
pip_install("insurance-elasticity==0.1.0", "pytest")

# COMMAND ----------

os.makedirs("/tmp/ie_repo", exist_ok=True)
if not os.path.exists("/tmp/ie_repo/tests"):
    clone = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/burning-cost/insurance-elasticity.git",
         "/tmp/ie_repo"],
        capture_output=True, text=True
    )
    print("Clone:", clone.returncode, clone.stderr[:300])

# COMMAND ----------

test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=long", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
full_out = test_result.stdout + "\n" + test_result.stderr
print(full_out[-10000:])

# COMMAND ----------

status = "PASSED" if test_result.returncode == 0 else "FAILED"
summary_lines = [l for l in full_out.split("\n") if "passed" in l or "failed" in l or "FAILED" in l or "ERROR" in l]
summary = "\n".join(summary_lines[-30:])
dbutils.notebook.exit(f"{status} rc={test_result.returncode}\n\nSUMMARY:\n{summary}\n\nOUTPUT (last 3500):\n{full_out[-3500:]}")
