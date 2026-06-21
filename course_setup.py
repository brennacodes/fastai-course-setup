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

import atexit
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
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
    # The autosave controller (see AutoSaver), or None when autosave is off. Notebooks
    # use it to explicitly persist a result, e.g. ``context.saver.keep(learn)``.
    saver: Optional["AutoSaver"] = None
    # The base Google Drive folder where artifacts are stored, or None if Drive is not
    # available (for example on a plain local machine).
    drive_dir: Optional[Path] = None
    # The auto-sync handle: a SyncDaemon on a machine with a git repo, a ColabExporter on
    # Colab, or None when auto-sync is off or has nothing to do.
    sync: Optional[object] = None


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


# ---------------------------------------------------------------------------
# Durable artifact persistence + as-you-go git history
#
# Colab runtimes are ephemeral: when one disconnects, everything on its local disk is
# wiped, including a model you just spent GPU time training. The helpers below save
# expensive results - trained models, scraped/cleaned image folders, downloaded tables -
# to Google Drive the moment they are produced, and record a small manifest describing
# each one. A separate ``snapshot`` step commits notebooks and those manifests to git on
# your own machine, so you also get a trustworthy history of your work.
#
# Two machines are involved: the notebook runs on Colab (which can reach Drive), while
# your git repo lives on your Mac. So the design is two halves bridged by Drive - an
# in-kernel saver (Colab) and a git snapshotter (your machine). As elsewhere in this
# module, nothing heavy is imported at the top: fastai, IPython and pandas are imported
# lazily, inside the functions that use them, so this stays import-safe and testable.
# ---------------------------------------------------------------------------

# Standard Colab mount point, checked first. On a Mac/PC the "Google Drive for Desktop"
# app mounts Drive elsewhere; _detect_drive_mount finds that too. Override everything with
# the FASTAI_DRIVE_ROOT environment variable if your Drive lives somewhere unusual.
DRIVE_MOUNTS = ("/content/drive/MyDrive",)

# Glob patterns for "My Drive" when Google Drive for Desktop is installed. The account name
# is part of the path, so we glob for it rather than hard-coding an address.
DRIVE_DESKTOP_GLOBS = (
    "~/Library/CloudStorage/GoogleDrive-*/My Drive",  # macOS
    "~/Google Drive/My Drive",
    "G:/My Drive",  # common Windows drive letter
)

# A cell whose source contains one of these is treated as "this cell trained a model", so we
# re-save the model afterwards. Training updates a learner's weights in place, so its object
# identity does not change and we cannot notice the update by identity alone - the verb is
# how we know to re-save. Matched as plain substrings, deliberately loose: a false positive
# only costs one extra cheap save.
TRAINING_VERBS = ("fine_tune", "fit_one_cycle", "fit_flat_cos", ".fit(")

# Object ids of models saved during the current cell. The post_run_cell hook checks this so
# it does not re-save a model that cached_model (or an explicit keep) already persisted in
# the same cell, which would otherwise create a duplicate .pkl and manifest entry.
_models_saved_this_cell = set()

# True once an AutoSaver has registered its post_run_cell hook (the hook is what clears the
# set above after each cell). save_model only records ids while this is on, so the set can
# never grow without bound when autosave is off and nothing would ever clear it.
_autosave_hook_active = False


def _persist_log(message: str) -> None:
    """Print a namespaced progress line so saves are visible and easy to grep."""
    print(f"[course_setup] {message}")


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string, e.g. '2026-06-19T21:15:02+00:00'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _human_size(path: Path) -> str:
    """Format a file size for humans, e.g. '45.1MB'."""
    size = float(Path(path).stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"  # unreachable in practice; keeps the return explicit


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 checksum, read in chunks so large files are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_drive_mount() -> Optional[Path]:
    """Find this machine's Google Drive 'My Drive' folder, or None if there isn't one.

    Tries Colab's mount first, then Google Drive for Desktop's CloudStorage location.
    Returning None lets callers degrade to a harmless no-op when Drive is unavailable.
    """
    for mount in DRIVE_MOUNTS:
        if Path(mount).exists():
            return Path(mount)
    for pattern in DRIVE_DESKTOP_GLOBS:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return Path(matches[0])
    return None


def drive_root() -> Optional[Path]:
    """Return the base Drive folder for course artifacts, or None if Drive isn't here.

    Checks the FASTAI_DRIVE_ROOT override first, then the detected Drive mount. Returning
    None (instead of raising) lets every caller degrade to a harmless no-op when there is
    no Drive, which is exactly what we want on a machine without it.
    """
    override = os.environ.get("FASTAI_DRIVE_ROOT")
    if override:
        return Path(override)
    mount = _detect_drive_mount()
    return mount / "fastai" if mount is not None else None


def artifacts_dir(lesson: str) -> Optional[Path]:
    """Return (creating it) the Drive folder holding one lesson's artifacts, or None."""
    root = drive_root()
    if root is None:
        return None
    directory = root / "artifacts" / lesson
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def manifest_path(lesson: str) -> Optional[Path]:
    """Path to a lesson's manifest file on Drive, or None when Drive is unavailable."""
    directory = artifacts_dir(lesson)
    return (directory / "artifact-manifest.json") if directory else None


def load_manifest(lesson: str) -> dict:
    """Read a lesson's manifest, returning an empty skeleton if none exists yet."""
    path = manifest_path(lesson)
    if path and path.exists():
        with open(path) as handle:
            return json.load(handle)
    return {"lesson": lesson, "artifacts": {}}


def record_artifact(lesson: str, name: str, entry: dict) -> Optional[Path]:
    """Merge one artifact's record into the lesson manifest and write it back.

    ``name`` is the manifest key, so re-saving an artifact updates its record in place
    rather than appending a duplicate. Returns the manifest path, or None without Drive.
    """
    path = manifest_path(lesson)
    if path is None:
        return None
    manifest = load_manifest(lesson)
    manifest["lesson"] = lesson
    manifest.setdefault("artifacts", {})[name] = entry
    manifest["updated_at"] = _utc_now_iso()
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def zip_dir(source_dir: Path, archive_path: Path) -> Path:
    """Zip an entire directory tree into a single archive file.

    Image datasets are hundreds of tiny files, and Google Drive's mounted filesystem is
    slow per-file. Copying one zip is far faster and more reliable than syncing each
    image, which is why datasets are stored on Drive as a single archive.
    """
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(source_dir.rglob("*")):
            if item.is_file():
                bundle.write(item, item.relative_to(source_dir))
    return archive_path


def unzip_dir(archive_path: Path, dest_dir: Path) -> Path:
    """Extract a zip archive into ``dest_dir`` (created if needed)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as bundle:
        bundle.extractall(dest_dir)
    return dest_dir


def _record_file(lesson, name, kind, file_path, source=None, extra=None) -> dict:
    """Build and store a manifest entry for one saved file. Returns the entry."""
    file_path = Path(file_path)
    entry = {
        "kind": kind,
        "drive_path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": _sha256(file_path),
        "saved_at": _utc_now_iso(),
    }
    if extra:
        entry.update(extra)  # kind-specific fields, e.g. a folder's restore location
    if source:
        entry.update(source)  # notebook + cell tagging, if known
    record_artifact(lesson, name, entry)
    return entry


def save_model(learner, lesson: str, name: str, source: Optional[dict] = None) -> Optional[Path]:
    """Export a fastai learner to Drive and record it. Returns the .pkl path, or None.

    Duck-typed: anything with an ``.export(path)`` method works, so this module never has
    to import fastai. ``learner.export`` writes the model together with its dataloaders,
    so it can be loaded back for inference later.
    """
    directory = artifacts_dir(lesson)
    if directory is None:
        _persist_log(f"no Drive available; model '{name}' not saved")
        return None
    models = directory / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / f"{name}.pkl"
    learner.export(target)
    if _autosave_hook_active:
        _models_saved_this_cell.add(id(learner))
    _record_file(lesson, name, "model", target, source)
    _persist_log(f"saved model '{name}' -> {target} ({_human_size(target)})")
    return target


def save_folder(folder, lesson: str, name: str, source: Optional[dict] = None) -> Optional[Path]:
    """Zip a data folder to Drive and record it. Returns the archive path, or None."""
    directory = artifacts_dir(lesson)
    if directory is None:
        _persist_log(f"no Drive available; dataset '{name}' not saved")
        return None
    archive = directory / "datasets" / f"{name}.zip"
    zip_dir(folder, archive)
    # Remember where this folder lived so restore() can put the images back exactly where
    # the notebook reads them (e.g. 'pill_data/pill_or_not'), not in some generic location.
    _record_file(lesson, name, "dataset", archive, source, extra={"restore_to": str(Path(folder))})
    _persist_log(f"saved dataset '{name}' -> {archive} ({_human_size(archive)})")
    return archive


def save_dataframe(frame, lesson: str, name: str, source: Optional[dict] = None) -> Optional[Path]:
    """Save a pandas DataFrame to Drive as CSV and record it. Returns the path, or None.

    CSV is used rather than parquet so this needs no extra library (parquet would require
    installing pyarrow). For the course's metadata tables that round-trips fine.
    """
    directory = artifacts_dir(lesson)
    if directory is None:
        _persist_log(f"no Drive available; table '{name}' not saved")
        return None
    target = directory / "datasets" / f"{name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    _record_file(lesson, name, "dataframe", target, source)
    _persist_log(f"saved table '{name}' -> {target} ({_human_size(target)})")
    return target


def save_file(file_path, lesson: str, name: str, source: Optional[dict] = None) -> Optional[Path]:
    """Copy a single file a cell produced to Drive and record it. Returns the path, or None.

    The original location is recorded so restore() can put it back where the notebook wrote
    it. This is what lets autosave persist arbitrary outputs (a saved .pkl, a CSV, an image)
    without knowing in advance what kind of thing they are.
    """
    directory = artifacts_dir(lesson)
    if directory is None:
        return None
    target = directory / "files" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, target)
    _record_file(lesson, name, "file", target, source, extra={"restore_to": str(Path(file_path))})
    _persist_log(f"saved file '{name}' -> {target} ({_human_size(target)})")
    return target


def cached_model(lesson, name, build_fn, load_fn=None, source=None):
    """Return an existing model from Drive if present, else build then save it.

    ``build_fn()`` trains and returns a learner; it only runs on a cache miss.
    ``load_fn(path)`` loads a saved model back - if you do not pass one, fastai's
    ``load_learner`` is imported lazily and used. This is the guard that turns a
    disconnect-and-retrain into a few-second reload.
    """
    directory = artifacts_dir(lesson)
    if directory is not None:
        existing = directory / "models" / f"{name}.pkl"
        if existing.exists():
            if load_fn is None:
                from fastai.learner import load_learner

                load_fn = load_learner
            _persist_log(f"loaded model '{name}' from Drive; skipped retraining")
            return load_fn(existing)
    learner = build_fn()
    save_model(learner, lesson, name, source)
    return learner


def cached_folder(lesson, name, build_fn, dest_dir, source=None):
    """Restore a dataset folder from Drive if present, else build then save it.

    ``build_fn()`` populates ``dest_dir`` (scrape/clean/download) and only runs on a cache
    miss. On a hit the saved archive is unzipped back into ``dest_dir``. Returns the
    folder path. This preserves slow, non-deterministic work like a DuckDuckGo scrape and
    your manual image cleaning.
    """
    dest_dir = Path(dest_dir)
    directory = artifacts_dir(lesson)
    if directory is not None:
        archive = directory / "datasets" / f"{name}.zip"
        if archive.exists():
            unzip_dir(archive, dest_dir)
            _persist_log(f"restored dataset '{name}' from Drive; skipped rebuild")
            return dest_dir
    build_fn()
    save_folder(dest_dir, lesson, name, source)
    return dest_dir


def cached_dataframe(lesson, name, build_fn, source=None):
    """Return a table from Drive (CSV) if present, else build then save it."""
    directory = artifacts_dir(lesson)
    if directory is not None:
        existing = directory / "datasets" / f"{name}.csv"
        if existing.exists():
            import pandas as pd

            _persist_log(f"loaded table '{name}' from Drive; skipped re-download")
            return pd.read_csv(existing, low_memory=False)
    frame = build_fn()
    save_dataframe(frame, lesson, name, source)
    return frame


def _looks_like_learner(obj) -> bool:
    """True for a fastai Learner: something we can both export and predict with."""
    return callable(getattr(obj, "export", None)) and callable(getattr(obj, "predict", None))


def _saveable_kind(obj) -> Optional[str]:
    """Classify a namespace object as a saveable artifact, or None to leave it alone.

    'model' for a fastai Learner (anything with export + predict); 'dataframe' for a pandas
    DataFrame (anything with to_csv + columns). Everything else - numbers, strings, plots,
    DataLoaders - returns None, so autosave persists results, not every variable that exists.
    """
    if _looks_like_learner(obj):
        return "model"
    if callable(getattr(obj, "to_csv", None)) and getattr(obj, "columns", None) is not None:
        return "dataframe"
    return None


# Directory names never scanned when looking for files a cell produced: version control,
# caches, virtualenvs, the mounted Drive itself, and Colab's bundled sample data. Hidden
# directories (names starting with ".") are skipped too.
AUTOSAVE_IGNORE_DIRS = {
    ".git", ".ipynb_checkpoints", "__pycache__", ".cache", ".config", ".local",
    "node_modules", ".venv", "venv", "env", "drive", "sample_data", ".fastai-course-setup",
}

# A guard so the after-cell scan stays fast even if the working tree is enormous.
AUTOSAVE_MAX_FILES = 20000


def _scan_tree(root: Path) -> dict:
    """Map file path -> modification time for files under root, skipping noise directories.

    Used to notice which files a cell created or changed so they can be mirrored to Drive.
    Caches, virtualenvs, the Drive mount and hidden folders are pruned so we only see the
    user's actual working files.
    """
    index = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in AUTOSAVE_IGNORE_DIRS and not name.startswith(".")
        ]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                index[path] = os.path.getmtime(path)
            except OSError:
                continue
            if len(index) >= AUTOSAVE_MAX_FILES:
                return index
    return index


def _get_ipython():
    """Return the active IPython shell, or None when not running under IPython/Jupyter."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _user_namespace() -> dict:
    """The notebook's variable namespace, or an empty dict outside a kernel."""
    ipython = _get_ipython()
    return ipython.user_ns if ipython is not None else {}


def current_notebook() -> Optional[str]:
    """Best-effort notebook filename for tagging artifacts; None if unknown.

    VS Code's Jupyter sets a global ``__vsc_ipynb_file__``; we also honour a
    ``FASTAI_NOTEBOOK_NAME`` override. Colab does not expose the name to the kernel, so
    this can legitimately return None, in which case artifacts are simply tagged
    'unknown'.
    """
    candidate = _user_namespace().get("__vsc_ipynb_file__") or os.environ.get(
        "FASTAI_NOTEBOOK_NAME"
    )
    return os.path.basename(str(candidate)) if candidate else None


class AutoSaver:
    """Ambient autosave: after every cell, save whatever that cell produced to Drive.

    Turned on by ``init(autosave=True, ...)`` and needs no per-cell code. It registers a
    callback on IPython's ``post_run_cell`` event - the signal behind the run-count and the
    success/error mark next to each cell - so it runs after every cell in any notebook. Each
    run it saves: (a) models and dataframes that appeared or changed in the notebook, and
    (b) any files or folders the cell wrote under the working directory. Everything is
    recorded in a per-lesson manifest, tagged with the notebook and cell that produced it.
    ``keep(...)`` / ``keep_folder(...)`` remain for forcing a save by hand, and the bin
    scripts (snapshot/restore) are the manual on-demand levers.
    """

    def __init__(self, lesson: str):
        self.lesson = lesson
        self._registered = False
        self._ignored = set()
        self._execution_count = 0
        # Working-tree snapshot (file path -> mtime); None until the first cell establishes
        # a baseline, so pre-existing files are not all re-saved on the first run.
        self._file_index = None
        # Variable name -> id of the object last saved under it, to notice new/changed ones.
        self._object_index = {}

    def register(self) -> "AutoSaver":
        """Subscribe to the cell-execution event. No-op (with a note) outside a kernel."""
        ipython = _get_ipython()
        if ipython is None:
            _persist_log("autosave inactive (not running inside IPython/Jupyter)")
            return self
        if drive_root() is None:
            _persist_log("autosave armed; it will save once Google Drive is mounted")
        ipython.events.register("post_run_cell", self._on_post_run_cell)
        global _autosave_hook_active
        _autosave_hook_active = True
        self._registered = True
        _persist_log(f"autosave on for lesson '{self.lesson}'")
        return self

    def ignore(self, name: str) -> "AutoSaver":
        """Never auto-save the artifact with this name."""
        self._ignored.add(name)
        return self

    def keep(self, obj, name: str = "model"):
        """Explicitly save a result now: a learner -> model, otherwise a table -> CSV."""
        if _looks_like_learner(obj):
            return save_model(obj, self.lesson, name, self._source())
        return save_dataframe(obj, self.lesson, name, self._source())

    def keep_folder(self, folder, name: str):
        """Explicitly zip and save a data folder now (e.g. after manual cleaning)."""
        return save_folder(folder, self.lesson, name, self._source())

    def _source(self, cell_source: Optional[str] = None) -> dict:
        """The notebook + cell tag attached to every artifact this saver writes."""
        info = {
            "notebook": current_notebook() or "unknown",
            "cell_execution_count": self._execution_count,
        }
        if cell_source is not None:
            info["cell_sha256"] = hashlib.sha256(
                cell_source.encode("utf-8")
            ).hexdigest()[:12]
        return info

    def _on_post_run_cell(self, result) -> None:
        """Run after every cell; save whatever artifacts it produced. Never raises."""
        try:
            self._execution_count += 1
            if getattr(result, "success", True) is False:
                return  # a cell that errored did not produce a trustworthy result
            cell_source = getattr(getattr(result, "info", None), "raw_cell", "") or ""
            self._save_namespace_objects(cell_source)
            self._save_new_files(cell_source)
        except Exception as error:  # autosave must never break the user's notebook
            _persist_log(f"autosave skipped a cell after an error: {error!r}")
        finally:
            # Reset the per-cell de-dup set so the next cell starts clean.
            _models_saved_this_cell.clear()

    def _save_namespace_objects(self, cell_source: str) -> None:
        """Save models / dataframes this cell created or (re)trained."""
        trained = any(verb in cell_source for verb in TRAINING_VERBS)
        for name, value in list(_user_namespace().items()):
            if name.startswith("_") or name in self._ignored:
                continue
            if id(value) in _models_saved_this_cell:
                continue  # already saved by cached_model/keep in this same cell
            kind = _saveable_kind(value)
            if kind is None:
                continue
            is_new = self._object_index.get(name) != id(value)
            if kind == "model" and trained:
                # Save a model only when a cell actually trained one, not when an untrained
                # learner is first constructed (that would waste tens of MB on empty weights).
                save_model(value, self.lesson, name, self._source(cell_source))
                self._object_index[name] = id(value)
            elif kind == "dataframe" and is_new:
                save_dataframe(value, self.lesson, name, self._source(cell_source))
                self._object_index[name] = id(value)

    def _save_new_files(self, cell_source: str) -> None:
        """Mirror files this cell created or changed under the working dir to Drive."""
        root = Path.cwd()
        current = _scan_tree(root)
        if self._file_index is None:
            # First cell: record what already existed so we don't re-save the whole tree.
            self._file_index = current
            return
        touched = {
            path for path in set(current) | set(self._file_index)
            if current.get(path) != self._file_index.get(path)
        }
        self._file_index = current
        if not touched:
            return
        # Group touched paths by their first directory under the working dir: loose files in
        # the working dir are saved individually, while a changed subdirectory (a dataset)
        # is zipped once as a whole.
        loose_files = []
        changed_subdirs = set()
        for path in touched:
            try:
                relative = Path(path).relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) == 1:
                loose_files.append(path)
            else:
                changed_subdirs.add(relative.parts[0])
        for path in loose_files:
            if Path(path).exists() and Path(path).name not in self._ignored:
                save_file(path, self.lesson, Path(path).name, self._source(cell_source))
        for subdir in changed_subdirs:
            folder = root / subdir
            if subdir not in self._ignored and folder.is_dir():
                save_folder(folder, self.lesson, subdir, self._source(cell_source))


def find_repo_root(start: Optional[str] = None) -> Optional[Path]:
    """Walk up from ``start`` (or the current directory) to the enclosing git repo."""
    current = Path(start or os.getcwd()).resolve()
    for directory in [current, *current.parents]:
        if (directory / ".git").exists():
            return directory
    return None


def _snapshot_pathspecs():
    """What snapshot is allowed to commit: notebooks and manifests only, never binaries.

    These are git pathspecs. A leading ``*`` matches across directories, so both patterns
    reach into every subfolder (notebooks anywhere, the per-lesson manifests under
    ``homework/<lesson>/``). Artifact binaries are git-ignored anyway; restricting the add
    here is a second belt-and-braces guard.
    """
    return ["*.ipynb", "*artifact-manifest.json"]


def _commit_once(repo_root: Path, message_prefix: str, quiet_when_clean: bool = False,
                 push: bool = True) -> bool:
    """Stage notebooks + manifests, commit if anything changed, and push. Returns True if so.

    When ``push`` is True (the default) a successful commit is followed by a best-effort
    ``git push`` so the work lands on the remote, not just in the local repo. The push can
    fail for reasons that have nothing to do with the commit (offline, no upstream, auth), so
    it never changes the return value or raises - see ``_push_to_remote``.
    """
    root = str(repo_root)
    for spec in _snapshot_pathspecs():
        # Add each pathspec on its own. A spec that matches nothing (for example, no
        # manifests have been created yet) makes ``git add`` print a harmless "did not
        # match" and exit non-zero; combining specs into one call would let that abort the
        # whole add, so we keep them separate and swallow that stderr.
        add_args = ["git", "-C", root, "add", "--", spec]
        if spec == "*.ipynb":
            # Never commit the scratch files the round-trip writes for unresolved merge
            # conflicts; they are local aids, not real notebooks.
            add_args.append(f":(exclude)*{NOTEBOOK_CONFLICT_SUFFIX}")
        subprocess.run(
            add_args,
            check=False,
            stderr=subprocess.DEVNULL,
        )
    # ``diff --cached --quiet`` exits 0 when nothing is staged, 1 when there are changes.
    staged = subprocess.run(["git", "-C", root, "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        if not quiet_when_clean:
            _persist_log("snapshot: nothing to commit")
        return False
    message = f"{message_prefix}: {_utc_now_iso()}"
    subprocess.run(["git", "-C", root, "commit", "-m", message], check=False)
    _persist_log(f"snapshot committed: {message}")
    if push:
        _push_to_remote(repo_root)
    return True


def _push_to_remote(repo_root: Path) -> bool:
    """Best-effort ``git push`` of the current branch. Returns True only if the push landed.

    A snapshot is only useful off your laptop once it reaches the remote, so we push right
    after committing. But a push can fail for reasons unrelated to your work - no network, no
    configured upstream, an auth prompt - and none of that should interrupt a notebook run or
    lose the commit (which is already safe locally). So every failure is logged and swallowed.
    """
    root = str(repo_root)
    result = subprocess.run(
        ["git", "-C", root, "push"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        _persist_log("snapshot pushed to remote")
        return True
    detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
    _persist_log(f"snapshot push skipped (commit is saved locally): {detail}")
    return False


def snapshot(repo_root=None, once: bool = True, interval: float = 5.0,
             message_prefix: str = "snapshot", push: bool = True) -> bool:
    """Commit notebooks + manifests to git and push. Returns True if a commit was made.

    Runs only where a git repo exists (your own machine); on Colab there is no repo, so
    this is a logged no-op. With ``once=False`` it polls and commits whenever files
    change, until interrupted - that is what ``bin/snapshot --auto`` runs. Running
    ``--once`` is independent: it commits and exits without touching any ``--auto`` loop.

    Each commit is followed by a best-effort ``git push`` so the remote stays current; pass
    ``push=False`` to commit without pushing (a failed push never loses the local commit).
    """
    root = find_repo_root(repo_root)
    if root is None:
        _persist_log("snapshot skipped: no git repository here (expected on Colab)")
        return False
    if once:
        return _commit_once(root, message_prefix, push=push)
    _persist_log("snapshot --auto watching for changes; press Ctrl+C to stop")
    try:
        while True:
            _commit_once(root, message_prefix, quiet_when_clean=True, push=push)
            time.sleep(interval)
    except KeyboardInterrupt:
        _persist_log("snapshot --auto stopped")
    return False


def sync(lesson: str, repo_root=None) -> bool:
    """Ensure this lesson's work is captured, then snapshot. Safe to call repeatedly.

    Artifacts are written to Drive at save time, so sync's remaining job is the git
    snapshot. On Colab that is a no-op (no repo) - your work is already safe on Drive;
    on your own machine it commits the notebook and manifest.
    """
    return snapshot(repo_root=repo_root, once=True)


def _notebook_paths(repo_root: Path):
    """Yield (absolute_path, path_relative_to_repo) for every notebook under repo_root.

    The same noise directories the autosaver skips are pruned (version control, caches,
    virtualenvs, the Drive mount, checkpoints), and hidden directories are skipped. Editor
    backup files such as ``something.ipynb.bak`` are naturally excluded because they do not
    end in ``.ipynb``.
    """
    repo_root = Path(repo_root)
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            name for name in dirnames
            if name not in AUTOSAVE_IGNORE_DIRS and not name.startswith(".")
        ]
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.endswith(NOTEBOOK_CONFLICT_SUFFIX):
                absolute = Path(dirpath) / filename
                yield absolute, absolute.relative_to(repo_root)


def _mirror_once(repo_root: Path, dest_base: Path, source_mtimes: Optional[dict] = None) -> int:
    """Copy notebooks whose content differs from their Drive copy. Returns the count copied.

    When ``source_mtimes`` is given (the ``--auto`` loop), a notebook is only re-examined if
    its modification time changed since the last pass, so an idle watcher does no work. The
    content is still compared by checksum before copying, so an unchanged file is never
    needlessly re-copied to the (slow) mounted Drive.
    """
    copied = 0
    for absolute, relative in _notebook_paths(repo_root):
        try:
            mtime = absolute.stat().st_mtime
        except OSError:
            continue
        unchanged_since_last_pass = (
            source_mtimes is not None and source_mtimes.get(str(absolute)) == mtime
        )
        if not unchanged_since_last_pass:
            destination = dest_base / relative
            if not destination.exists() or _sha256(absolute) != _sha256(destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(absolute, destination)
                _persist_log(f"mirrored notebook '{relative}' -> {destination}")
                copied += 1
        if source_mtimes is not None:
            source_mtimes[str(absolute)] = mtime
    _write_path_index(repo_root, dest_base)
    return copied


def mirror_notebooks(repo_root=None, once: bool = True, interval: float = 5.0) -> int:
    """Copy the repo's notebooks to Drive as you edit, for a faithful off-laptop copy.

    The in-kernel autosaver saves a cell's *results* to Drive, but it cannot save the
    notebook *file*: on Colab the kernel only ever receives a cell's source when that cell
    runs, never the ``.ipynb`` document, which your local editor owns. This Mac-side mirror
    closes that gap. It copies each notebook into ``<drive>/notebooks/<relative path>`` so a
    faithful copy lands in the same Google Drive account as your artifacts.

    Run it once per session. Like ``snapshot --auto`` it is an independent mtime-poll loop,
    so a manual ``--once`` run never disturbs a running ``--auto`` watcher (they are separate
    processes). Returns the number copied on a single pass (``once=True``); the watcher loop
    returns 0 when interrupted. It is a logged no-op where there is no git repo (Colab) or no
    Drive, mirroring how ``snapshot``/``restore`` degrade.
    """
    root = find_repo_root(repo_root)
    if root is None:
        _persist_log("mirror skipped: no git repository here (expected on Colab)")
        return 0
    drive = drive_root()
    if drive is None:
        _persist_log("mirror skipped: no Drive available")
        return 0
    dest_base = drive / "notebooks"
    if once:
        copied = _mirror_once(root, dest_base)
        _persist_log(f"mirror: copied {copied} notebook(s) to {dest_base}")
        return copied
    _persist_log("mirror --auto watching notebooks; press Ctrl+C to stop")
    source_mtimes: dict = {}
    try:
        while True:
            _mirror_once(root, dest_base, source_mtimes)
            time.sleep(interval)
    except KeyboardInterrupt:
        _persist_log("mirror --auto stopped")
    return 0


def restore(lesson: str, dest_root=None) -> int:
    """Copy a lesson's artifacts from Drive back into the working tree. Returns the count.

    ``dest_root`` is the base directory restored paths are resolved against, and defaults to
    the current working directory. That default is what makes the in-notebook resume flow
    "just work": the notebook saved its datasets at cwd-relative paths (e.g.
    ``pill_data/pill_or_not``), so restoring relative to the same cwd puts them back exactly
    where the notebook reads them - on Colab (cwd ``/content``) and locally alike. The
    ``bin/restore`` wrapper passes an explicit ``dest_root`` so a terminal run lands the data
    in the lesson's directory regardless of where the wrapper is invoked from. The manifest
    itself is also copied into ``dest_root`` so git can track it.
    """
    directory = artifacts_dir(lesson)
    if directory is None:
        _persist_log("restore skipped: no Drive available")
        return 0
    dest_root = Path(dest_root) if dest_root else Path.cwd()
    manifest = load_manifest(lesson)
    restored = 0
    for name, entry in manifest.get("artifacts", {}).items():
        source = Path(entry["drive_path"])
        if not source.exists():
            # The manifest knows about it, but the file is not where it was saved - usually
            # Drive is mounted at a different root than when it was written. Warn rather
            # than silently undercount.
            _persist_log(f"restore: skipping '{name}' (not found at {source})")
            continue
        # Put things back where the notebook wrote them, using the path recorded at save
        # time. A relative recorded path is resolved against dest_root; an absolute one is
        # used as-is; with no recorded path we fall back to a generic per-kind location.
        kind = entry.get("kind")
        recorded = entry.get("restore_to")
        if recorded:
            candidate = Path(recorded)
            recorded_target = candidate if candidate.is_absolute() else dest_root / candidate
        else:
            recorded_target = None

        if kind == "dataset" and source.suffix == ".zip":
            unzip_dir(source, recorded_target or dest_root / "datasets" / name)
        elif kind == "file" and recorded_target is not None:
            recorded_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, recorded_target)
        else:
            subdir = "models" if kind == "model" else "datasets"
            target = dest_root / subdir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        restored += 1
    manifest_file = manifest_path(lesson)
    if manifest_file and manifest_file.exists():
        repo_manifest = dest_root / "artifact-manifest.json"
        repo_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_file, repo_manifest)
    _persist_log(f"restored {restored} artifact(s) for '{lesson}' into {dest_root}")
    return restored


# ---------------------------------------------------------------------------
# Notebook round-trip: bring Colab edits home, with a notebook-aware 3-way merge
# ---------------------------------------------------------------------------
#
# ``mirror`` pushes the Mac's notebooks out to Drive. ``pull_notebooks`` closes the loop: a
# notebook a Colab session exported to Drive is reconciled back into the repo. When only one
# side changed since they last agreed, the newer content simply wins. When BOTH changed, we do
# a notebook-aware 3-way merge with nbdime, using the last-synced copy (kept on Drive under
# ``notebooks/.sync-base``) as the common ancestor. Clean merges apply automatically; genuine
# conflicts are written to a ``<name>.merge-conflict.ipynb`` scratch file and left for you to
# resolve by hand, with both original sides preserved and the base untouched.

NOTEBOOK_CONFLICT_SUFFIX = ".merge-conflict.ipynb"
SYNC_BASE_DIRNAME = ".sync-base"


def _require_notebook_tools():
    """Import nbformat + nbdime, or return (None, None) with a logged note if unavailable.

    Both are imported lazily so the module loads anywhere; the round-trip only needs them on
    the Mac. nbformat ships with Jupyter; nbdime is the one extra dependency this adds.
    """
    try:
        import nbformat
        import nbdime
        return nbformat, nbdime
    except Exception as error:
        _persist_log(f"notebook merge unavailable (need nbformat + nbdime): {error!r}")
        return None, None


def _read_notebook_node(path: Path):
    """Parse a notebook file into an nbformat node, or None if it cannot be read."""
    nbformat, _ = _require_notebook_tools()
    if nbformat is None:
        return None
    try:
        return nbformat.read(str(path), as_version=4)
    except Exception as error:
        _persist_log(f"could not read notebook {path}: {error!r}")
        return None


def _notebook_fingerprint(path: Path) -> Optional[str]:
    """A normalized string of a notebook's content for change detection, or None.

    Two files with the same cells but trivially different on-disk formatting (key order, a
    trailing newline) normalize to the same string, so a plain re-save is not mistaken for an
    edit. Returns None when the file is missing or unreadable.
    """
    if not path.exists():
        return None
    nbformat, _ = _require_notebook_tools()
    if nbformat is None:
        return None
    node = _read_notebook_node(path)
    if node is None:
        return None
    try:
        return nbformat.writes(node)
    except Exception:
        return None


def _write_notebook_node(node, path: Path) -> None:
    """Write an nbformat node to ``path``, creating parent folders as needed."""
    nbformat, _ = _require_notebook_tools()
    if nbformat is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(node, str(path))


def _copy_notebook(source: Path, dest: Path) -> None:
    """Copy a notebook file verbatim, creating parent folders as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _sync_base_path(notebooks_dir: Path, relative: Path) -> Path:
    """Where the last-agreed copy of a notebook lives on Drive (the merge ancestor)."""
    return notebooks_dir / SYNC_BASE_DIRNAME / relative


def _conflict_scratch_path(mac_path: Path) -> Path:
    """The local ``<name>.merge-conflict.ipynb`` scratch file for a notebook."""
    return mac_path.parent / (mac_path.stem + NOTEBOOK_CONFLICT_SUFFIX)


def _drive_notebook_relpaths(notebooks_dir: Path):
    """Relative paths of real notebooks under ``<drive>/notebooks`` (no base, no scratch)."""
    results = []
    if not notebooks_dir.exists():
        return results
    for dirpath, dirnames, filenames in os.walk(notebooks_dir):
        dirnames[:] = [d for d in dirnames if d != SYNC_BASE_DIRNAME and not d.startswith(".")]
        for filename in filenames:
            if filename.endswith(".ipynb") and not filename.endswith(NOTEBOOK_CONFLICT_SUFFIX):
                absolute = Path(dirpath) / filename
                results.append(absolute.relative_to(notebooks_dir))
    return results


def pull_notebooks(repo_root=None, once: bool = True, interval: float = 5.0) -> int:
    """Reconcile notebooks between the Mac repo and Drive, merging when both sides changed.

    Returns the number of notebooks whose Mac copy was created or updated on a single pass, so
    a caller knows whether a snapshot is worth taking. A logged no-op where there is no repo
    (Colab) or no Drive. With ``once=False`` it polls until interrupted, like the snapshot and
    mirror watchers.
    """
    root = find_repo_root(repo_root)
    if root is None:
        _persist_log("pull skipped: no git repository here (expected on Colab)")
        return 0
    drive = drive_root()
    if drive is None:
        _persist_log("pull skipped: no Drive available")
        return 0
    if _require_notebook_tools()[1] is None:
        return 0
    notebooks_dir = drive / "notebooks"
    if once:
        changed = _pull_once(root, notebooks_dir)
        _persist_log(f"pull: updated {changed} notebook(s) from Drive")
        return changed
    _persist_log("pull --auto watching Drive notebooks; press Ctrl+C to stop")
    try:
        while True:
            _pull_once(root, notebooks_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        _persist_log("pull --auto stopped")
    return 0


def _pull_once(repo_root: Path, notebooks_dir: Path) -> int:
    """One reconciliation pass over the union of notebooks on the Mac and on Drive."""
    _write_path_index(repo_root, notebooks_dir)
    relpaths = {relative for _, relative in _notebook_paths(repo_root)}
    relpaths.update(_drive_notebook_relpaths(notebooks_dir))
    changed = 0
    for relative in sorted(relpaths, key=str):
        if _sync_one_notebook(repo_root, notebooks_dir, relative):
            changed += 1
    return changed


def _sync_one_notebook(repo_root: Path, notebooks_dir: Path, relative: Path) -> bool:
    """Reconcile one notebook across Mac, Drive, and the stored base.

    Returns True only when the Mac copy was created or changed - the signal that a snapshot is
    worth taking.
    """
    mac_path = repo_root / relative
    drive_path = notebooks_dir / relative
    base_path = _sync_base_path(notebooks_dir, relative)

    mac_fingerprint = _notebook_fingerprint(mac_path)
    drive_fingerprint = _notebook_fingerprint(drive_path)
    base_fingerprint = _notebook_fingerprint(base_path)

    # Crucial safety check: a file that is present on disk but momentarily unreadable (an editor
    # is mid-save, or a Drive sync landed a partial file) must NOT be treated as absent. If we
    # did, the "exists on only one side" branches below would happily copy the other side over
    # it and destroy a real edit. So if either live copy exists but will not parse, skip this
    # notebook for now; a later pass will pick it up once the write has settled.
    if (mac_path.exists() and mac_fingerprint is None) or (
        drive_path.exists() and drive_fingerprint is None
    ):
        _persist_log(f"pull: skipping '{relative}' this pass (a copy is unreadable right now)")
        return False

    mac_exists = mac_fingerprint is not None
    drive_exists = drive_fingerprint is not None

    # A new local notebook not yet on Drive: seed both Drive and the base from the Mac.
    if mac_exists and not drive_exists:
        _copy_notebook(mac_path, drive_path)
        _copy_notebook(mac_path, base_path)
        return False
    # A new notebook arrived from Colab: bring it down and seed the base.
    if drive_exists and not mac_exists:
        _copy_notebook(drive_path, mac_path)
        _copy_notebook(drive_path, base_path)
        return True
    if not mac_exists and not drive_exists:
        return False

    mac_changed = base_fingerprint is None or mac_fingerprint != base_fingerprint
    drive_changed = base_fingerprint is None or drive_fingerprint != base_fingerprint

    if not mac_changed and not drive_changed:
        return False
    if mac_changed and not drive_changed:
        _copy_notebook(mac_path, drive_path)
        _copy_notebook(mac_path, base_path)
        return False
    if drive_changed and not mac_changed:
        _copy_notebook(drive_path, mac_path)
        _copy_notebook(drive_path, base_path)
        return True
    return _merge_notebook(mac_path, drive_path, base_path)


def _merge_notebook(mac_path: Path, drive_path: Path, base_path: Path) -> bool:
    """3-way merge a notebook that changed on both sides. Returns True if the Mac copy changed.

    A clean merge is written to all three locations (Mac, Drive, base) so the two sides are
    back in agreement. A genuine conflict overwrites nothing: it leaves both sides in place,
    keeps the base unchanged so the conflict is re-detected until resolved, and drops a
    ``<name>.merge-conflict.ipynb`` scratch file holding nbdime's best-effort merge.
    """
    nbformat, nbdime = _require_notebook_tools()
    if nbdime is None:
        return False
    base = _read_notebook_node(base_path)
    mac = _read_notebook_node(mac_path)
    drive = _read_notebook_node(drive_path)
    conflict_path = _conflict_scratch_path(mac_path)

    # Without a readable common ancestor we cannot do a real 3-way merge, so we must not pick
    # one side over the other. Treat it as a conflict and leave both in place.
    if base is None or mac is None or drive is None:
        _persist_log(
            f"pull: cannot merge '{mac_path.name}' (missing base/local/remote); left both "
            f"sides in place. Colab's copy is at {drive_path}."
        )
        return False

    merged, decisions = nbdime.merge_notebooks(base, mac, drive)
    conflicts = [decision for decision in decisions if decision.get("conflict")]
    if not conflicts:
        _write_notebook_node(merged, mac_path)
        _write_notebook_node(merged, drive_path)
        _write_notebook_node(merged, base_path)
        if conflict_path.exists():
            conflict_path.unlink()
        _persist_log(f"pull: auto-merged '{mac_path.name}' (no conflicts)")
        return True

    _write_notebook_node(merged, conflict_path)
    _persist_log(
        f"pull: CONFLICT merging '{mac_path.name}'. Your Mac copy is unchanged; Colab's copy "
        f"is at {drive_path}; a merge attempt is at {conflict_path.name}. Resolve it, save "
        f"over {mac_path.name}, then run `bin/sync` (or course_setup resolve)."
    )
    return False


def resolve_conflict(notebook_path, repo_root=None) -> bool:
    """Accept a hand-resolved notebook as the new truth: update Drive + base, drop the scratch.

    After you fix a conflicted notebook (editing the real ``.ipynb``), call this to make the
    Mac version authoritative again. It copies your resolved notebook to Drive, advances the
    merge base to match, and deletes the ``<name>.merge-conflict.ipynb`` scratch file. Returns
    True on success.
    """
    root = find_repo_root(repo_root)
    if root is None:
        _persist_log("resolve skipped: no git repository here")
        return False
    drive = drive_root()
    if drive is None:
        _persist_log("resolve skipped: no Drive available")
        return False
    mac_path = Path(notebook_path)
    if not mac_path.is_absolute():
        mac_path = root / mac_path
    if not mac_path.exists():
        _persist_log(f"resolve: notebook not found: {mac_path}")
        return False
    notebooks_dir = drive / "notebooks"
    relative = mac_path.relative_to(root)
    _copy_notebook(mac_path, notebooks_dir / relative)
    _copy_notebook(mac_path, _sync_base_path(notebooks_dir, relative))
    conflict_path = _conflict_scratch_path(mac_path)
    if conflict_path.exists():
        conflict_path.unlink()
    _persist_log(f"resolve: '{mac_path.name}' is now the agreed version on Mac and Drive")
    return True


PATH_INDEX_FILENAME = ".path-index.json"
# Landing folder for notebooks exported from Colab that the Mac has never seen (so they are not
# in the path index). They are brought into the repo here, for you to move to their real home.
COLAB_ORIGIN_DIRNAME = "_from-colab"


def _path_index_file(notebooks_dir: Path) -> Path:
    return notebooks_dir / PATH_INDEX_FILENAME


def _write_path_index(repo_root: Path, notebooks_dir: Path) -> None:
    """Record basename -> [repo-relative paths] on Drive so a Colab export can place itself.

    The Colab kernel knows only a notebook's filename, not where it lives in the repo. The Mac
    does, so each mirror/pull pass writes this small index to Drive. Export reads it to turn
    'pill-or-not.ipynb' back into 'homework/lesson-1/side-quest/pill-or-not.ipynb'.
    """
    index: dict = {}
    for _, relative in _notebook_paths(repo_root):
        index.setdefault(Path(relative).name, []).append(str(relative))
    try:
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        _path_index_file(notebooks_dir).write_text(json.dumps(index, indent=2, sort_keys=True))
    except Exception as error:
        _persist_log(f"could not write notebook path index: {error!r}")


def _read_path_index(notebooks_dir: Path) -> dict:
    path = _path_index_file(notebooks_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_export_relpath(notebooks_dir: Path, basename: str) -> Optional[str]:
    """The unique repo-relative path for a notebook basename, or None if not safely resolvable.

    Returning None on a missing or ambiguous name is deliberate: it is safer to skip the export
    and tell the user than to drop a notebook at the wrong place in the tree.
    """
    matches = _read_path_index(notebooks_dir).get(basename, [])
    if len(matches) == 1:
        return matches[0]
    return None


def export_notebook_to_drive() -> bool:
    """Best-effort: write the live Colab notebook's JSON to Drive. Returns True on success.

    The Colab kernel cannot normally see its own ``.ipynb`` file, but Colab exposes the live
    document through an internal message channel. We ask for it and write it to
    ``<drive>/notebooks/<repo path>``, where the Mac-side ``pull`` reconciles it. This relies on
    Colab internals that can change without notice, so every failure is logged loudly and
    swallowed - it must never break a user's notebook run, and it never guesses a path it cannot
    confirm (which would misplace the file).
    """
    drive = drive_root()
    if drive is None:
        _persist_log("export skipped: no Drive available")
        return False
    try:
        from google.colab import _message
    except Exception:
        _persist_log("export skipped: not running on Colab (no google.colab._message)")
        return False
    try:
        reply = _message.blocking_request("get_ipynb", timeout_sec=30)
        notebook_json = reply["ipynb"]
    except Exception as error:
        _persist_log(
            "export FAILED: could not read the live notebook from Colab "
            f"({error!r}). Nothing is lost - save or download the notebook by hand."
        )
        return False
    notebooks_dir = drive / "notebooks"
    raw_name = current_notebook()
    basename = Path(raw_name).name if raw_name else None
    if not basename:
        _persist_log("export FAILED: could not determine the notebook filename on Colab.")
        return False
    relative = _resolve_export_relpath(notebooks_dir, basename)
    if relative is None:
        # Not in the Mac's path index: the notebook was probably created on Colab (the Mac has
        # never seen it) or its name is ambiguous. Rather than guess a path and misplace it, or
        # drop the work, land it in a clearly-named folder; the Mac-side pull brings it into the
        # repo there, and you can move it to its real home.
        relative = str(Path(COLAB_ORIGIN_DIRNAME) / basename)
        _persist_log(
            f"export: '{basename}' is not in the Drive path index; writing it under "
            f"'{COLAB_ORIGIN_DIRNAME}/' so it is not lost. Move it to its real path on the Mac."
        )
    nbformat, _ = _require_notebook_tools()
    if nbformat is None:
        return False
    try:
        node = nbformat.reads(json.dumps(notebook_json), as_version=4)
        _write_notebook_node(node, notebooks_dir / relative)
    except Exception as error:
        _persist_log(f"export FAILED: could not write notebook to Drive ({error!r})")
        return False
    _persist_log(f"export: wrote live notebook to {notebooks_dir / relative}")
    return True


def _default_lesson() -> str:
    """Tag to use when ``init`` is called with autosave on but no explicit lesson name."""
    name = current_notebook()
    return os.path.splitext(name)[0] if name else "lesson"


# ---------------------------------------------------------------------------
# Auto-sync: keep work flowing between Mac, Drive, and git without manual steps
# ---------------------------------------------------------------------------
#
# ``init(auto_sync=True)`` wires this up. On a machine with a git repo (your Mac) a background
# ``SyncDaemon`` runs the full loop on a timer: pull Colab edits home, mirror notebooks to
# Drive, then commit and push. On Colab there is no repo, so instead a ``ColabExporter`` pushes
# the live notebook to Drive as you run cells, and the Mac's daemon completes the round-trip.
# Only one handle is ever started per kernel.

_auto_sync_handle = None


def _sync_lock_path(repo_root: Path) -> Path:
    """The lock file marking 'a syncer is already running for this repo'."""
    return Path(repo_root) / ".git" / "course-setup-sync.lock"


def _process_alive(pid: int) -> bool:
    """True if a process with this pid currently exists (best-effort, cross-platform-ish)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, just owned by someone else
    except OSError:
        return False
    return True


def _acquire_sync_lock(repo_root: Path) -> Optional[Path]:
    """Take the repo's sync lock, or return None if a live syncer already holds it.

    This is what lets the in-kernel daemon and a terminal ``bin/sync --auto`` coexist: whoever
    starts first wins, and the other backs off instead of double-committing. A lock left behind
    by a dead process (stale pid) is reclaimed.
    """
    lock_path = _sync_lock_path(repo_root)
    if not lock_path.parent.exists():
        return lock_path  # no .git dir to lock in; nothing else could be running either
    try:
        handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        return lock_path
    except FileExistsError:
        try:
            holder = int((lock_path.read_text().strip() or "0"))
        except Exception:
            holder = 0
        # A lock held by ANY live process - including this one - means a syncer is already
        # active, so back off. Only a lock left by a dead process (stale pid) is reclaimed.
        if holder and _process_alive(holder):
            return None
        try:
            lock_path.write_text(str(os.getpid()))
            return lock_path
        except Exception:
            return None
    except Exception:
        # If locking itself misbehaves, do not block syncing - just run unlocked.
        return lock_path


def _release_sync_lock(lock_path: Optional[Path]) -> None:
    """Delete a sync lock if we still own it."""
    if lock_path is None:
        return
    try:
        if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
            lock_path.unlink()
    except Exception:
        pass


class SyncDaemon:
    """A background timer that reconciles notebooks and commits, where a git repo exists.

    Each pass runs pull -> mirror -> snapshot in order (never concurrently), so the git and
    file operations do not race each other. The worker is a daemon thread, so it never blocks
    the notebook process from exiting, and any single failing pass is logged and skipped rather
    than killing the loop.
    """

    def __init__(self, repo_root, interval: float = 5.0):
        self.repo_root = Path(repo_root)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._lock_path = None

    def _actions(self):
        return (
            ("pull", lambda: pull_notebooks(repo_root=self.repo_root, once=True)),
            ("mirror", lambda: mirror_notebooks(repo_root=self.repo_root, once=True)),
            ("snapshot", lambda: snapshot(repo_root=self.repo_root, once=True)),
        )

    def run_once(self) -> None:
        """Run one pull -> mirror -> snapshot pass, isolating each step's failures."""
        for label, action in self._actions():
            try:
                action()
            except Exception as error:
                _persist_log(f"auto-sync {label} pass failed: {error!r}")

    def _run(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval)

    def start(self) -> "SyncDaemon":
        self._lock_path = _acquire_sync_lock(self.repo_root)
        if self._lock_path is None:
            _persist_log(
                "auto-sync: another syncer is already running for this repo; not starting a second"
            )
            return self
        self._thread = threading.Thread(target=self._run, name="course-setup-sync", daemon=True)
        self._thread.start()
        # The worker is a daemon thread, so it is killed abruptly at interpreter exit without
        # running its own cleanup. Release the lock via atexit so a clean kernel shutdown frees
        # it promptly instead of leaving a stale lock for the next run to reclaim.
        atexit.register(_release_sync_lock, self._lock_path)
        _persist_log("auto-sync: background snapshot/mirror/pull daemon started")
        return self

    def stop(self) -> None:
        self._stop.set()
        _release_sync_lock(self._lock_path)
        self._lock_path = None


class ColabExporter:
    """On Colab, push the live notebook to Drive after cells run (throttled).

    There is no git repo on Colab, so the round-trip's Colab half is simply getting the edited
    notebook file onto Drive; the Mac daemon does the rest. We hang this off IPython's
    ``post_run_cell`` event like autosave does, but throttle it so we are not serializing the
    whole notebook on every single cell.
    """

    def __init__(self, min_interval: float = 20.0):
        self.min_interval = min_interval
        self._last_export = 0.0
        self._registered = False

    def register(self) -> "ColabExporter":
        ipython = _get_ipython()
        if ipython is None:
            _persist_log("colab export inactive (not running inside IPython/Jupyter)")
            return self
        ipython.events.register("post_run_cell", self._on_post_run_cell)
        # The post-cell throttle means edits in the last seconds before a Colab runtime
        # disconnects might never be pushed. A best-effort export at interpreter exit flushes
        # that final state so a clean shutdown does not lose the last few minutes of work.
        atexit.register(self._flush)
        self._registered = True
        _persist_log("colab export armed; the live notebook will sync to Drive as cells run")
        return self

    def _flush(self) -> None:
        try:
            export_notebook_to_drive()
        except Exception as error:
            _persist_log(f"colab export flush error: {error!r}")

    def _on_post_run_cell(self, result=None) -> None:
        now = time.monotonic()
        if now - self._last_export < self.min_interval:
            return
        self._last_export = now
        try:
            export_notebook_to_drive()
        except Exception as error:  # the hook must never break the user's notebook
            _persist_log(f"colab export hook error: {error!r}")


def _start_auto_sync(env: Environment):
    """Start the right auto-sync half for this machine. Returns its handle, or None.

    Idempotent: only one handle is created per kernel, so re-running ``init`` does not pile up
    daemons or hooks.
    """
    global _auto_sync_handle
    if _auto_sync_handle is not None:
        return _auto_sync_handle
    repo = find_repo_root()
    if repo is not None:
        _auto_sync_handle = SyncDaemon(repo).start()
    elif env.in_colab or env.iscolab:
        _auto_sync_handle = ColabExporter().register()
    else:
        _persist_log("auto-sync: nothing to do (no git repo here, not on Colab)")
    return _auto_sync_handle


def run_sync(repo_root=None, once: bool = True, interval: float = 5.0) -> None:
    """Foreground pull -> mirror -> snapshot, for ``bin/sync``. A no-op without a git repo.

    ``--once`` runs a single pass and exits. ``--auto`` loops until interrupted, taking the
    repo's sync lock first so it cooperates with (rather than fights) an in-kernel daemon that
    a running notebook may already have started.
    """
    root = find_repo_root(repo_root)
    if root is None:
        _persist_log("sync skipped: no git repository here (expected on Colab)")
        return
    daemon = SyncDaemon(root, interval=interval)
    if once:
        daemon.run_once()
        return
    lock_path = _acquire_sync_lock(root)
    if lock_path is None:
        _persist_log("sync --auto: another syncer is already running for this repo; exiting")
        return
    _persist_log("sync --auto: pull + mirror + snapshot on a loop; press Ctrl+C to stop")
    try:
        while True:
            daemon.run_once()
            time.sleep(interval)
    except KeyboardInterrupt:
        _persist_log("sync --auto stopped")
    finally:
        _release_sync_lock(lock_path)


def init(
    packages: Sequence[str] = ("fastai",),
    setup_book: bool = False,
    competition: Optional[str] = None,
    mount_drive: bool = True,
    wide_print: bool = False,
    internet_check: bool = False,
    autosave: bool = True,
    auto_sync: bool = True,
    lesson: Optional[str] = None,
) -> SetupContext:
    """Run the standard course setup and return a context with the results.

    Steps, in a safe order: detect the environment, optionally check internet, optionally
    mount Drive, install the requested packages, optionally run ``fastbook.setup_book()``,
    optionally download a Kaggle competition, optionally widen print output, and finally
    pick the device. Packages are installed before the device is selected because device
    selection needs ``torch`` (which fastai/fastbook bring in).

    When ``autosave`` is on (the default), an :class:`AutoSaver` is wired up last so that
    expensive results - trained models, and any data folders you ``watch`` - are saved to
    Google Drive as they are produced and recorded in a manifest. ``lesson`` is the name
    those artifacts are filed under; if omitted it is guessed from the notebook filename.
    Autosave is lazy and harmless: it does nothing until Drive is available and a real
    artifact appears, so leaving it on in a notebook that never trains costs nothing.

    When ``auto_sync`` is on (the default), the notebook file itself is kept flowing too. On a
    machine with a git repo (your Mac) a background daemon pulls Colab edits home, mirrors
    notebooks to Drive, and commits + pushes on a timer. On Colab, where there is no repo, the
    live notebook is pushed to Drive as cells run so the Mac daemon can complete the loop. Pass
    ``auto_sync=False`` to opt out (for example, if you prefer to run ``bin/sync`` by hand).
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

    saver = AutoSaver(lesson or _default_lesson()).register() if autosave else None
    sync_handle = _start_auto_sync(env) if auto_sync else None

    return SetupContext(
        device=device,
        in_colab=env.in_colab,
        iskaggle=env.iskaggle,
        iscolab=env.iscolab,
        path=path,
        saver=saver,
        drive_dir=drive_root(),
        sync=sync_handle,
    )


def _main(argv=None):
    """Command-line entry point used by ``bin/snapshot`` and ``bin/restore``.

    Examples:
        python course_setup.py snapshot --once
        python course_setup.py snapshot --auto
        python course_setup.py mirror --auto
        python course_setup.py restore lesson-1
    """
    import argparse

    parser = argparse.ArgumentParser(description="course_setup persistence commands")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="commit notebooks + manifests to git")
    mode = snap.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true", help="watch and commit continuously")
    mode.add_argument("--once", action="store_true", help="commit once and exit (default)")

    mir = sub.add_parser("mirror", help="copy notebooks to Drive as you edit")
    mir_mode = mir.add_mutually_exclusive_group()
    mir_mode.add_argument("--auto", action="store_true", help="watch and mirror continuously")
    mir_mode.add_argument("--once", action="store_true", help="mirror once and exit (default)")

    rest = sub.add_parser("restore", help="copy a lesson's artifacts from Drive into the repo")
    rest.add_argument("lesson", help="lesson name, e.g. lesson-1")
    rest.add_argument(
        "--dest",
        default=None,
        help="base directory to restore into (defaults to the current directory)",
    )

    pull_parser = sub.add_parser("pull", help="bring Colab notebook edits back from Drive")
    pull_mode = pull_parser.add_mutually_exclusive_group()
    pull_mode.add_argument("--auto", action="store_true", help="watch and reconcile continuously")
    pull_mode.add_argument("--once", action="store_true", help="reconcile once and exit (default)")

    resolve_parser = sub.add_parser(
        "resolve", help="accept a hand-resolved notebook after a merge conflict"
    )
    resolve_parser.add_argument("notebook", help="path to the resolved .ipynb")

    sync_parser = sub.add_parser(
        "sync", help="pull + mirror + snapshot in one go (the full round-trip)"
    )
    sync_mode = sync_parser.add_mutually_exclusive_group()
    sync_mode.add_argument("--auto", action="store_true", help="run the loop continuously")
    sync_mode.add_argument("--once", action="store_true", help="run one pass and exit (default)")

    args = parser.parse_args(argv)
    if args.command == "snapshot":
        snapshot(once=not args.auto)
    elif args.command == "mirror":
        mirror_notebooks(once=not args.auto)
    elif args.command == "restore":
        restore(args.lesson, dest_root=args.dest)
    elif args.command == "pull":
        pull_notebooks(once=not args.auto)
    elif args.command == "resolve":
        resolve_conflict(args.notebook)
    elif args.command == "sync":
        run_sync(once=not args.auto)


if __name__ == "__main__":
    _main()
