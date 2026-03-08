# Databricks notebook source
# Runs the full test suite for insurance-elasticity.

# COMMAND ----------

import subprocess, sys, os

def pip_install(*args):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + list(args),
        capture_output=True, text=True
    )
    out = result.stdout[-2000:]
    err = result.stderr[-2000:]
    if result.returncode != 0:
        raise RuntimeError(f"pip failed: {args}\n{out}\n{err}")
    return out

print("Python:", sys.version)
pip_install("econml==0.15.1")
print("econml OK")
pip_install("catboost>=1.2")
print("catboost OK")
pip_install("pandas>=2.0", "polars>=0.20")
print("pandas+polars OK")
pip_install("insurance-elasticity==0.1.0", "pytest")
print("insurance-elasticity OK")

# COMMAND ----------

os.makedirs("/tmp/ie_repo", exist_ok=True)
if not os.path.exists("/tmp/ie_repo/tests"):
    clone = subprocess.run(
        ["git", "clone", "--depth=1",
         "https://github.com/burning-cost/insurance-elasticity.git",
         "/tmp/ie_repo"],
        capture_output=True, text=True
    )
    if clone.returncode != 0:
        raise RuntimeError(f"clone failed: {clone.stderr}")

# COMMAND ----------

# Reproduce the session fixture manually to get the full traceback
debug = subprocess.run(
    [sys.executable, "-c", """
import traceback
import sys
sys.path.insert(0, "src")
from insurance_elasticity.data import make_renewal_data
from insurance_elasticity.fit import RenewalElasticityEstimator

CONFOUNDERS = ["age", "ncd_years", "vehicle_group", "channel"]
try:
    print("Making data...")
    df = make_renewal_data(n=500, seed=42)
    print("Data made, shape:", df.shape)
    est = RenewalElasticityEstimator(
        cate_model="causal_forest",
        n_estimators=10,
        catboost_iterations=20,
        n_folds=2,
        random_state=42,
    )
    print("Fitting...")
    est.fit(df, confounders=CONFOUNDERS)
    print("Fit OK")
    ate = est.ate()
    print("ATE:", ate)
except Exception as e:
    print("EXCEPTION:", type(e).__name__, str(e))
    traceback.print_exc()
"""],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
print("STDOUT:", debug.stdout[-5000:])
print("STDERR:", debug.stderr[-3000:])

# COMMAND ----------

# Run tests with full traceback on session fixture error
test_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_fit.py::TestFitBasic::test_fit_returns_self",
     "-v", "--tb=long", "--no-header"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
full_out = test_result.stdout + "\n" + test_result.stderr
print(full_out[-12000:])

# COMMAND ----------

all_result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/tmp/ie_repo",
)
all_out = all_result.stdout + "\n" + all_result.stderr
status = "PASSED" if all_result.returncode == 0 else "FAILED"
summary_lines = [l for l in all_out.split("\n") if any(k in l for k in ("passed", "failed", "FAILED"))]
summary = "\n".join(summary_lines[-10:])
dbutils.notebook.exit(
    f"{status} rc={all_result.returncode}\n\nDEBUG STDOUT:\n{debug.stdout[-2000:]}\n\nFIT TEST (last 4000):\n{full_out[-4000:]}\n\nSUMMARY:\n{summary}"
)
