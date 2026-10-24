"""
scripts/build_clean_lockfile.py
Phase 6 Part 1.2:
Creates a clean venv OUTSIDE the project, installs requirements-deploy.txt (+ pytest, httpx),
runs the tests, and saves its pip freeze as requirements-deploy.lock.txt (source of truth).
"""

import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"[*] Running: {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed with exit code {res.returncode}")
    return res

def main():
    repo_root = Path(__file__).resolve().parent.parent
    base_python = Path(sys.base_prefix) / "python.exe"
    if not base_python.exists():
        base_python = Path(sys.executable)

    # Temporary venv location strictly OUTSIDE the project
    clean_venv_dir = Path(tempfile.gettempdir()) / "depi_clean_deploy_venv"
    if clean_venv_dir.exists():
        print(f"[*] Removing existing clean venv at {clean_venv_dir}...")
        shutil.rmtree(clean_venv_dir, ignore_errors=True)

    print(f"[*] Creating clean venv at: {clean_venv_dir}")
    run_cmd([str(base_python), "-m", "venv", str(clean_venv_dir)])

    clean_python = clean_venv_dir / "Scripts" / "python.exe"
    clean_pip = clean_venv_dir / "Scripts" / "pip.exe"

    print("[*] Upgrading pip in clean venv...")
    run_cmd([str(clean_python), "-m", "pip", "install", "--no-cache-dir", "--upgrade", "pip"])

    reqs_path = repo_root / "requirements-deploy.txt"
    print(f"[*] Installing {reqs_path} (+ test runners pytest, httpx) into clean venv...")
    run_cmd([str(clean_python), "-m", "pip", "install", "--no-cache-dir", "-r", str(reqs_path), "pytest", "httpx"])

    print("[*] Running full test suite from clean venv...")
    test_res = run_cmd([str(clean_python), "-m", "pytest", "tests", "-v"], cwd=repo_root)
    print("[+] All tests PASSED in clean venv!")

    print("[*] Generating requirements-deploy.lock.txt (source of truth)...")
    freeze_res = run_cmd([str(clean_python), "-m", "pip", "freeze"])
    
    lock_path = repo_root / "requirements-deploy.lock.txt"
    lock_path.write_text(freeze_res.stdout, encoding="utf-8")
    print(f"[+] Saved freeze output to {lock_path} ({len(freeze_res.stdout.splitlines())} pinned packages)")

    print("[+] Part 1.2 clean venv verification and lockfile generation complete!")

if __name__ == "__main__":
    main()
