"""
scripts/run_notebook.py
Executes notebooks/01_flight_delay_eda_to_training.ipynb using nbclient
and saves the executed notebook with all outputs and figures embedded.
"""

import time
import nbformat
from nbclient import NotebookClient

def execute_notebook():
    nb_path = "notebooks/01_flight_delay_eda_to_training.ipynb"
    print(f"[*] Reading notebook from {nb_path}...")
    nb = nbformat.read(nb_path, as_version=4)
    
    print("[*] Starting notebook execution with NotebookClient...")
    t0 = time.time()
    client = NotebookClient(nb, timeout=1200, kernel_name="python3")
    client.execute()
    elapsed = time.time() - t0
    print(f"[+] Notebook executed successfully in {elapsed:.1f} seconds!")
    
    print(f"[*] Writing executed notebook back to {nb_path}...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("[+] Done! All cell outputs and figures are persisted in the notebook.")

if __name__ == "__main__":
    execute_notebook()
