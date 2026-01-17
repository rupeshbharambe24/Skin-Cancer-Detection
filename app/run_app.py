import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "vision-studio-backend"
FRONTEND_DIR = ROOT / "vision-studio-ui"


def run_process(command, cwd):
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=os.environ.copy(),
    )


def terminate_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    npm_exe = "npm.cmd" if os.name == "nt" else "npm"
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    frontend_cmd = [npm_exe, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]

    backend_proc = run_process(backend_cmd, BACKEND_DIR)
    try:
        frontend_proc = run_process(frontend_cmd, FRONTEND_DIR)
    except FileNotFoundError:
        terminate_process(backend_proc)
        print("Failed to start frontend. Ensure Node.js and npm are installed and on PATH.")
        print("Try: npm --version")
        sys.exit(1)

    print("Backend: http://127.0.0.1:8000")
    print("Frontend: http://127.0.0.1:5173")
    print("Press Ctrl+C to stop both.")

    try:
        while True:
            if backend_proc.poll() is not None or frontend_proc.poll() is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        terminate_process(frontend_proc)
        terminate_process(backend_proc)


if __name__ == "__main__":
    main()
