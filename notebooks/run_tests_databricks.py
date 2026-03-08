# Databricks notebook source
# Runs the full test suite for insurance-elasticity.
# Install dependencies step by step for better error visibility.

# COMMAND ----------
# MAGIC %md ## Install core dependencies

# COMMAND ----------

# Install heavy deps first so we can see any errors
# econml and catboost can take 3-5 minutes to install
import subprocess, sys

def run_pip(args):
    result = subprocess.run(
        [sys.executable, "-m", "pip"] + args,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-3000:])
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"pip failed: {' '.join(args)}")
    return result.stdout

# Install in stages
print("Installing econml...")
run_pip(["install", "econml>=0.15", "--quiet"])
print("Done.")

print("Installing catboost...")
run_pip(["install", "catboost>=1.2", "--quiet"])
print("Done.")

print("Installing insurance-elasticity and pytest...")
run_pip(["install", "insurance-elasticity==0.1.0", "pytest", "--quiet"])
print("Done.")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys, os

# Clone the repo to get the tests
os.makedirs("/tmp/ie_repo", exist_ok=True)
result = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burning-cost/insurance-elasticity.git",
     "/tmp/ie_repo"],
    capture_output=True, text=True
)
print("Clone stdout:", result.stdout)
print("Clone stderr:", result.stderr)
if result.returncode != 0:
    raise RuntimeError("git clone failed")
print("Clone OK.")

# COMMAND ----------

# Run the tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
output = result.stdout
if len(output) > 12000:
    output = output[-12000:]
print(output)
if result.stderr:
    stderr = result.stderr[-2000:]
    print("STDERR:", stderr)
print("Return code:", result.returncode)

# COMMAND ----------

if result.returncode == 0:
    print("ALL TESTS PASSED")
else:
    raise Exception(f"Tests FAILED with return code {result.returncode}")
