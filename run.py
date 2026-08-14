#!/usr/bin/env python
"""One entry point for everything in this project.

    python run.py                 show this list
    python run.py backend         start the API server
    python run.py frontend        start the web UI
    python run.py demo            recognition check in an OpenCV window
    python run.py train           retrain the classifier
    python run.py collect K       record training samples for a letter
    python run.py dataset         dataset health report
    python run.py signs           regenerate the sign illustrations
    python run.py check           verify the Gemini setup
    python run.py test            run the test suite
    python run.py doctor          check the install and report problems

It works from any directory and with any Python: if it is not already running
inside this project's virtual environment it re-launches itself there, so
``python run.py backend`` does the right thing even from a system Python that
has none of the dependencies installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Make sure we are running inside the project's virtual environment
# ----------------------------------------------------------------------


def venv_python() -> Path | None:
    """The interpreter inside .venv, if it exists."""
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",   # Windows
        ROOT / ".venv" / "bin" / "python",           # macOS / Linux
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / "venv" / "bin" / "python",
    ]
    return next((path for path in candidates if path.exists()), None)


def reexec_in_venv() -> None:
    """Re-run this script with the project interpreter, once."""
    if os.environ.get("HANDSIGN_REEXEC"):
        return  # already tried; do not loop

    interpreter = venv_python()
    if interpreter is None:
        return
    if Path(sys.executable).resolve() == interpreter.resolve():
        return  # already the right one

    print(f"[run] switching to {interpreter}", flush=True)
    environment = {**os.environ, "HANDSIGN_REEXEC": "1"}
    raise SystemExit(
        subprocess.call([str(interpreter), str(Path(__file__)), *sys.argv[1:]],
                        env=environment)
    )


reexec_in_venv()

# Importable no matter where the script was launched from.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


def _run(command: list[str], cwd: Path | None = None) -> int:
    print(f"[run] {' '.join(command)}")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    try:
        return subprocess.call(command, cwd=str(cwd or ROOT), env=env)
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        print(f"[run] command not found: {command[0]}", file=sys.stderr)
        return 1


def cmd_backend(args: list[str]) -> int:
    print("Starting the API on http://127.0.0.1:8000")
    print("Then, in a second terminal: python run.py frontend\n")
    return _run(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app",
         "--host", "127.0.0.1", "--port", "8000", "--reload", *args]
    )


def cmd_frontend(args: list[str]) -> int:
    if not (ROOT / "frontend" / "node_modules").exists():
        print("[run] installing frontend dependencies (first run only) ...")
        if _run(["npm", "install"], cwd=ROOT / "frontend") != 0:
            return 1
    print("\nOpen http://localhost:5173 once Vite is ready.")
    print("Use localhost, not 127.0.0.1 - browsers only allow camera access there.\n")
    npm = "npm.cmd" if os.name == "nt" else "npm"
    return _run([npm, "run", "dev", *args], cwd=ROOT / "frontend")


def cmd_demo(args: list[str]) -> int:
    return _run([sys.executable, "-m", "ml.scripts.live_demo", *args])


def cmd_train(args: list[str]) -> int:
    return _run([sys.executable, "-m", "ml.scripts.train_model", *args])


def cmd_collect(args: list[str]) -> int:
    if args and len(args[0]) == 1 and args[0].isalpha():
        args = ["--label", args[0].upper(), *args[1:]]
    return _run([sys.executable, "-m", "ml.scripts.collect_data", *args])


def cmd_dataset(args: list[str]) -> int:
    return _run([sys.executable, "-m", "ml.scripts.inspect_dataset", *args])


def cmd_signs(args: list[str]) -> int:
    return _run([sys.executable, "-m", "ml.scripts.generate_sign_assets", *args])


def cmd_check(args: list[str]) -> int:
    return _run([sys.executable, "-m", "backend.app.scripts.check_gemini", *args])


def cmd_test(args: list[str]) -> int:
    return _run([sys.executable, "-m", "pytest", *args])


def cmd_doctor(_args: list[str]) -> int:
    """Report what is and is not ready, instead of failing at the first step."""
    from importlib.util import find_spec

    ok = True
    print(f"python        : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        print("                MediaPipe needs Python 3.10-3.12.")
        ok = False

    print("\npackages")
    for module, hint in [
        ("mediapipe", "hand detection"),
        ("cv2", "image decoding (opencv-python)"),
        ("sklearn", "the classifier"),
        ("fastapi", "the API"),
        ("uvicorn", "the server"),
        ("langchain_google_genai", "Gemini suggestions (optional)"),
    ]:
        found = find_spec(module) is not None
        print(f"  {'ok ' if found else 'MISSING'}  {module:<24} {hint}")
        if not found and module != "langchain_google_genai":
            ok = False

    print("\nfiles")
    for path, hint in [
        (ROOT / "models" / "sign_model.pkl", "trained classifier"),
        (ROOT / "models" / "hand_landmarker.task", "MediaPipe model"),
        (ROOT / "dataset", "training data"),
        (ROOT / "frontend" / "node_modules", "frontend deps (npm install)"),
        (ROOT / "frontend" / "public" / "signs" / "A.svg", "sign illustrations"),
    ]:
        found = path.exists()
        print(f"  {'ok ' if found else 'MISSING'}  {path.name:<24} {hint}")
        if not found:
            ok = False

    print("\nconfiguration")
    env_file = ROOT / ".env"
    print(f"  {'ok ' if env_file.exists() else '-  '}  .env"
          f"{'' if env_file.exists() else ' (optional; local suggestions only)'}")

    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
        print(f"  ok    model      {settings.resolved_model_path().name}")
        print(f"  {'ok ' if settings.gemini_enabled else '-  '}    gemini     "
              f"{settings.gemini_model if settings.gemini_enabled else 'not configured'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED to load settings: {exc}")
        ok = False

    if not ok:
        print("\nSomething is missing. Fix:")
        print("  pip install -r requirements.txt")
        print("  cd frontend && npm install")
        return 1
    print("\nEverything is ready. Start with: python run.py backend")
    return 0


COMMANDS = {
    "backend": cmd_backend,
    "api": cmd_backend,
    "frontend": cmd_frontend,
    "ui": cmd_frontend,
    "demo": cmd_demo,
    "train": cmd_train,
    "collect": cmd_collect,
    "dataset": cmd_dataset,
    "signs": cmd_signs,
    "check": cmd_check,
    "test": cmd_test,
    "doctor": cmd_doctor,
}

#: Old file names, so the original commands give directions instead of
#: "No such file or directory".
RETIRED = {
    "live_prediction.py": "demo",
    "camera.py": "demo",
    "hand_tracking.py": "demo",
    "test_live_features.py": "demo",
    "train_model.py": "train",
    "data_collection.py": "collect",
    "inspect_dataset.py": "dataset",
    "test_features.py": "test",
    "fix_A_columns.py": None,
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    name = argv[0]
    if name in RETIRED:
        replacement = RETIRED[name]
        print(f"'{name}' was replaced during the rewrite.")
        if replacement:
            print(f"Use:  python run.py {replacement}")
        else:
            print("It was a one-time dataset fix and is no longer needed.")
        return 1

    handler = COMMANDS.get(name)
    if handler is None:
        print(f"Unknown command: {name}\n")
        print(__doc__)
        return 1
    return handler(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
