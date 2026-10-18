"""Reproduce the experiments; each stage runs in a fresh process."""
import os
import subprocess
import sys

if __name__=="__main__":
    env=dict(os.environ, PYTHONHASHSEED="42", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    for module in ["prepare","representations","classify","verify"]:
        subprocess.run([sys.executable,"-u","-m",f"experiment.{module}"],env=env,check=True)
