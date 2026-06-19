"""Shared setup for the fast.ai course notebooks.

Every notebook in the course needs the same boilerplate before the real work starts:
pick the best available GPU, figure out whether we are on Colab/Kaggle/local, install
the libraries the lesson needs, and (on Colab) mount Google Drive. This module holds
that logic in one place so the notebooks do not each repeat it.

Design notes worth knowing as you read this:

* Only the standard library is imported at the top of this file. Heavy libraries like
  ``torch`` and ``fastai`` are imported *inside* the functions that use them. That keeps
  this module safe to import anywhere - including on a plain machine with no GPU stack -
  which is what lets the tests run without installing the whole deep-learning toolchain.

* The notebooks still keep their own ``from fastai.vision.all import *`` (or
  ``from fastbook import *``) line. That is deliberate: Python's ``import *`` only adds
  names to the namespace where it runs. If this module ran it inside a function, those
  names would land in the function and vanish - they would never reach your notebook. So
  this module's job is to guarantee the package is *installed*; the star-import stays in
  the notebook.
"""

import importlib.util
import os
import re
import socket
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

# Let unimplemented Apple Metal (MPS) operations fall back to the CPU instead of raising.
# Setting this at import time means it is in place before torch is ever imported, which is
# when torch reads it. os.environ is part of Python's standard library.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class Environment:
    """Where this notebook is running. A dataclass is just a tidy struct of named fields."""

    in_colab: bool
    iskaggle: bool
    iscolab: bool


@dataclass
class SetupContext:
    """What ``init`` hands back to the notebook so it can use the results of setup."""

    device: object
    in_colab: bool
    iskaggle: bool
    iscolab: bool
    path: Optional[Path] = None


def detect_env() -> Environment:
    """Detect Colab/Kaggle/local from import probes and environment variables.

    ``in_colab`` is true when the ``google.colab`` package can be imported (it only
    exists inside Colab); this is the reliable "are we on Colab at all" signal.
    ``iskaggle`` and ``iscolab`` read the ``KAGGLE_KERNEL_RUN_TYPE`` and ``COLAB_GPU``
    environment variables, matching what the original course notebooks checked.
    """
    try:
        import google.colab  # noqa: F401  (imported only to test availability)

        in_colab = True
    except ImportError:
        in_colab = False

    iskaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""))
    iscolab = bool(os.environ.get("COLAB_GPU", ""))
    return Environment(in_colab=in_colab, iskaggle=iskaggle, iscolab=iscolab)


def _probe_name(spec: str) -> str:
    """Strip a pip version/extra spec down to the importable module name.

    ``"dtreeviz==1.4.1"`` -> ``"dtreeviz"``, ``"fastai"`` -> ``"fastai"``. We split on the
    first version or extras character so the install probe checks the real module name.
    """
    return re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0]


def ensure_packages(packages: Sequence[str]) -> None:
    """Install each package in ``packages`` only if it is not already importable.

    ``importlib.util.find_spec`` checks whether Python can find a module without actually
    importing it, so probing is cheap. We install with ``subprocess.run`` rather than the
    notebook ``!pip`` shell-magic because ``!`` only works inside notebooks - a plain
    ``.py`` module cannot use it. ``[sys.executable, "-m", "pip", ...]`` runs pip from the
    exact Python interpreter that is running this code, which avoids installing into the
    wrong environment.
    """
    for spec in packages:
        probe = _probe_name(spec)
        try:
            already_present = importlib.util.find_spec(probe) is not None
        except (ImportError, ValueError):
            already_present = False
        if already_present:
            print(f"{spec} is already installed")
        else:
            print(f"Installing {spec} ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-Uqq", spec],
                check=False,
            )


def mount_colab_drive(env: Optional[Environment] = None) -> bool:
    """Mount Google Drive when running on Colab. Returns True if a mount happened.

    Off Colab this is a harmless no-op, so notebooks can always ask for it.
    """
    env = env or detect_env()
    if not env.in_colab:
        return False
    from google.colab import drive

    drive.mount("/content/drive")
    return True


def check_internet() -> None:
    """Fail fast with a clear message if there is no internet (mainly for Kaggle).

    Kaggle requires phone verification before a notebook can reach the internet. This
    opens a tiny socket to a public DNS server; if it cannot connect, it raises with
    instructions instead of letting a later download fail in a confusing way.
    """
    try:
        socket.setdefaulttimeout(1)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 53))
    except OSError as error:
        raise RuntimeError(
            "STOP: No internet. On Kaggle, click '>|' in the top right "
            "and set the 'Internet' switch to on."
        ) from error


def select_device():
    """Pick the best available device (CUDA, then Apple MPS, then CPU) and tell fastai.

    Returns a ``torch.device``. ``torch`` is imported here, lazily, so this module stays
    importable on machines without it. If fastai is installed we also set it as fastai's
    default device so the rest of the lesson uses the GPU automatically.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple GPU)")
    else:
        device = torch.device("cpu")
        if detect_env().in_colab:
            print(
                "No GPU available; using CPU. "
                "You appear to be on Colab without a GPU runtime - "
                "enable it via Runtime > Change runtime type > Hardware accelerator > GPU, "
                "then re-run this cell."
            )
        else:
            print("No GPU available; using CPU")

    try:
        from fastai.torch_core import default_device

        default_device(device)
    except ImportError:
        pass

    return device


def download_competition(name: str, env: Optional[Environment] = None) -> Path:
    """Return the local path to a Kaggle competition's data, downloading it if needed.

    On Kaggle the data is already mounted under ``../input/<name>``. Elsewhere we use the
    Kaggle API to download and unzip it into a local folder named after the competition.
    """
    env = env or detect_env()
    if env.iskaggle:
        return Path(f"../input/{name}")

    path = Path(name)
    if not path.exists():
        import kaggle

        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f"{path}.zip").extractall(path)
    return path


def set_wide_print() -> None:
    """Widen numpy/torch/pandas console output so tables and tensors are easier to read."""
    import numpy as np
    import pandas as pd
    import torch

    np.set_printoptions(linewidth=140)
    torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
    pd.set_option("display.width", 140)


def init(
    packages: Sequence[str] = ("fastai",),
    setup_book: bool = False,
    competition: Optional[str] = None,
    mount_drive: bool = True,
    wide_print: bool = False,
    internet_check: bool = False,
) -> SetupContext:
    """Run the standard course setup and return a context with the results.

    Steps, in a safe order: detect the environment, optionally check internet, optionally
    mount Drive, install the requested packages, optionally run ``fastbook.setup_book()``,
    optionally download a Kaggle competition, optionally widen print output, and finally
    pick the device. Packages are installed before the device is selected because device
    selection needs ``torch`` (which fastai/fastbook bring in).
    """
    env = detect_env()

    if internet_check:
        check_internet()
    if mount_drive:
        mount_colab_drive(env)

    ensure_packages(packages)

    if setup_book:
        import fastbook

        fastbook.setup_book()

    path = download_competition(competition, env) if competition else None

    if wide_print:
        set_wide_print()

    device = select_device()

    return SetupContext(
        device=device,
        in_colab=env.in_colab,
        iskaggle=env.iskaggle,
        iscolab=env.iscolab,
        path=path,
    )
