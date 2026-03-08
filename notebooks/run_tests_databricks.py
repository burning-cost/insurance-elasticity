# Databricks notebook source
# Runs the full test suite for insurance-elasticity.

# COMMAND ----------

import subprocess, sys

def run_pip(args, verbose=True):
    result = subprocess.run(
        [sys.executable, "-m", "pip"] + args,
        capture_output=True, text=True
    )
    if verbose or result.returncode != 0:
        if result.stdout:
            print("STDOUT:", result.stdout[-5000:])
        if result.stderr:
            print("STDERR:", result.stderr[-5000:])
    return result.returncode, result.stdout, result.stderr

# Check existing packages first
print("=== Python version ===")
print(sys.version)

print("\n=== Checking econml ===")
rc, out, err = run_pip(["install", "econml>=0.15", "--dry-run"], verbose=True)
print(f"dry-run return code: {rc}")

# COMMAND ----------

print("=== Installing econml (this can take 5 minutes) ===")
rc, out, err = run_pip(["install", "econml>=0.15", "-v"], verbose=True)
print(f"econml install return code: {rc}")

# COMMAND ----------

print("=== Installing catboost ===")
rc, out, err = run_pip(["install", "catboost>=1.2", "--quiet"], verbose=True)
print(f"catboost install return code: {rc}")

# COMMAND ----------

print("=== Installing insurance-elasticity ===")
rc, out, err = run_pip(["install", "insurance-elasticity==0.1.0", "pytest", "--quiet"], verbose=True)
print(f"install return code: {rc}")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys, os

os.makedirs("/tmp/ie_repo", exist_ok=True)
result = subprocess.run(
    ["git", "clone", "--depth=1",
     "https://github.com/burning-cost/insurance-elasticity.git",
     "/tmp/ie_repo"],
    capture_output=True, text=True
)
print("Clone:", result.returncode, result.stdout, result.stderr)

# COMMAND ----------

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
out = result.stdout
if len(out) > 12000:
    out = out[-12000:]
print(out)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)

# COMMAND ----------

if result.returncode == 0:
    print("ALL TESTS PASSED")
else:
    raise Exception(f"Tests FAILED with return code {result.returncode}")
