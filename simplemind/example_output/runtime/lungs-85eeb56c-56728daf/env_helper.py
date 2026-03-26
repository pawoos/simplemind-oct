import os
import subprocess
from pathlib import Path
import yaml
import sys
import hashlib

# ----------------------
# Load Configuration
# ----------------------
def load_config(path="env.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

ENV_NAME = cfg["env_name"]
PYTHON_VERSION = cfg.get("python_version", "3.11")
CHANNELS = cfg.get("channels", [])
CONDA_PACKAGES = cfg.get("conda_packages", [])
PIP_PACKAGES = cfg.get("pip_packages", [])
REPO = cfg.get("repo", {})
USE_GPU = cfg.get("use_gpu", None)  # optional flag


# ----------------------
# Helper Functions
# ----------------------
def run(cmd, check=True):
    """Run a shell command with flush + safety."""
    print(f"[CMD] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=check)

# MM = "micromamba"
# ENV_NAME = "myenv"
# ENV_FILE = Path("env.yaml")
# HASH_FILE = Path(f".{ENV_NAME}_envhash")

# --- compute hash of env.yaml ---
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def env_exists(env_name: str) -> bool:
    """Check if micromamba env already exists."""
    result = subprocess.run(["micromamba", "env", "list"], capture_output=True, text=True)
    return env_name in result.stdout


def create_env(hash_dir):
    """Create the base Python environment if it doesn't exist."""
    env_created = False
    env_file = Path("env.yaml")
    new_hash = file_hash(env_file)

    env_hash_file = hash_dir / f".{ENV_NAME}_envhash"
    print(f"[INFO] environment hash file: {env_hash_file.resolve()}", flush=True)

    if not env_exists(ENV_NAME):
        env_hash_file.write_text(new_hash)

        print(f"[INFO] Creating environment '{ENV_NAME}' (Python {PYTHON_VERSION})...", flush=True)
        run(["micromamba", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"])
        env_created = True
    else:
        # --- compare with stored hash ---
        old_hash = env_hash_file.read_text().strip() if env_hash_file.exists() else None
        # print(f"[INFO] old_hash = {old_hash}", flush=True)

        if new_hash != old_hash:
            print(f"[INFO] Rebuilding {ENV_NAME} environment (env.yaml changed)...", flush=True)
            subprocess.run(["micromamba", "env", "update", "-n", ENV_NAME, "-f", str(env_file)], check=True)
            env_hash_file.write_text(new_hash)
            env_created = True
        else:
            print(f"[INFO] Environment {ENV_NAME} is up to date.", flush=True)
        # print(f"[INFO] Environment '{ENV_NAME}' already exists.", flush=True)
    return  env_created


def install_conda_packages():
    """Install conda dependencies using micromamba."""
    if not CONDA_PACKAGES:
        print("[INFO] No conda packages to install.", flush=True)
        return

    print("[INFO] Installing conda packages...", flush=True)
    cmd = ["micromamba", "install", "-n", ENV_NAME, "-y"]
    cmd.extend(CONDA_PACKAGES)

    # Add channels at the end (mamba channel precedence = left→right)
    for ch in CHANNELS:
        cmd.extend(["-c", ch])
    run(cmd)


def install_pip_packages():
    """Install pip dependencies inside the micromamba env."""
    if not PIP_PACKAGES:
        print("[INFO] No pip packages to install.", flush=True)
        return

    print("[INFO] Installing pip packages...", flush=True)
    run(["micromamba", "run", "-n", ENV_NAME, "pip", "install", *PIP_PACKAGES])


def clone_repo():
    """Clone and optionally editable-install a repo."""
    if not REPO:
        return

    url = REPO["url"]
    dest = Path(REPO["dest"]).expanduser().resolve()
    if not dest.exists():
        print(f"[INFO] Cloning repository {url} -> {dest}", flush=True)
        run(["git", "clone", url, str(dest)])
    else:
        print(f"[INFO] Repo already exists at {dest}", flush=True)

    if REPO.get("editable_install", False):
        print(f"[INFO] Editable installing {dest}", flush=True)
        run(["micromamba", "run", "-n", ENV_NAME, "pip", "install", "-e", str(dest)])


# ----------------------
# Unified setup_env
# ----------------------
def setup_env(hash_dir):
    """
    Unified environment setup supporting GPU/CPU, repo cloning, and pip/conda deps.
    Compatible with older env.yaml files (no use_gpu key required).
    """
    env_created = create_env(hash_dir)

    # Determine GPU/CPU mode
    use_gpu = USE_GPU
    if use_gpu is None:
        use_gpu = os.environ.get("TOTALSEG_USE_GPU", "1") == "1"

    # Inject GPU/CPU package conditionally
    if use_gpu:
        if not any("pytorch-cuda" in pkg for pkg in CONDA_PACKAGES):
            CONDA_PACKAGES.append("pytorch-cuda=11.8")
        print("[INFO] Configured for GPU (CUDA 11.8)", flush=True)
    else:
        if not any("cpuonly" in pkg for pkg in CONDA_PACKAGES):
            CONDA_PACKAGES.append("cpuonly")
        print("[INFO] Configured for CPU-only mode", flush=True)

    if env_created:
        install_conda_packages()
        install_pip_packages()
        clone_repo()
    # print(f"\n[INFO] Environment '{ENV_NAME}' setup complete.", flush=True)


def call_in_env(
    input_data: bytes,
    script_name: str,
    hash_dir: Path,
    script_args: list[str] | None = None,
    env_name: str = "default_env",
    setup_env_func=None,
):
    """
    Run a Python script in its own micromamba environment with arbitrary arguments.

    Parameters:
        input_data (bytes): Data to send to the subprocess via stdin.
        script_name (str): Name or path of the Python file to run (e.g., "ms2.py").
        hash_dir (Path): Path to the directory for he env hash file.
        script_args (list[str], optional): List of arguments to pass to the script.
        env_name (str): Name of the micromamba environment.
        setup_env_func (callable, optional): Function to run setup if env/lib missing.
        auto_setup (bool): Whether to auto-run setup if environment or libs missing.

    Returns:
        bytes: Raw stdout output from the subprocess.
    """
    if script_args is None:
        script_args = []

    # if lib_dir is None:
    #     lib_dir = Path("./lib")

    if setup_env_func is not None:
        print("[INFO] Running setup...", file=sys.stdout, flush=True)
        setup_env_func(hash_dir=hash_dir)
        print("[INFO] Setup complete.", file=sys.stdout, flush=True)

    # --- Build and run command ---
    cmd = ["micromamba", "run", "-n", env_name, "python", script_name] + script_args
    print(f"[INFO] Running command: {' '.join(cmd)}", file=sys.stdout, flush=True)

    result = subprocess.run(
        cmd,
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Call to {script_name} failed (code {result.returncode}):\n{result.stderr.decode()}"
        )

    return result.stdout


# ----------------------
# Entry point (optional)
# ----------------------
if __name__ == "__main__":
    setup_env()
