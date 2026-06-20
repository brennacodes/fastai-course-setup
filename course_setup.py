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

# A cell that ran one of these is treated as "this cell trained a model". These are matched
# as plain substrings against the cell's source, which is deliberately loose: a false
# positive only costs an extra (cheap) save of a learner-like object, while a false
# negative would silently fail to protect a trained model, which is the outcome we care
# about avoiding.
TRAINING_VERBS = ("fine_tune", "fit_one_cycle", "fit_flat_cos", ".fit(")
# A cell that ran one of these is treated as "this cell built/changed a dataset folder".
DATA_VERBS = ("download_images", "resize_images", "make_folder_dataset", "untar_data")

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


def detect_models_to_save(cell_source: str, namespace: dict):
    """Decide which trained models a just-run cell produced. Pure and side-effect free.

    Given the cell's source text and the notebook namespace, return the list of
    ``(variable_name, learner)`` pairs worth saving now. The rule: if the cell ran a
    training call, every learner-like object currently in the namespace is worth saving
    (re-saving simply overwrites with the latest weights). Being a pure function makes
    this trivial to unit test without a real kernel or fastai.
    """
    if not any(verb in cell_source for verb in TRAINING_VERBS):
        return []
    found = []
    for name, value in list(namespace.items()):
        if name.startswith("_"):
            continue
        if _looks_like_learner(value):
            found.append((name, value))
    return found


def _folder_signature(path: Path):
    """A cheap (file count, total bytes) fingerprint used to notice a folder changed."""
    path = Path(path)
    if not path.exists():
        return (0, 0)
    files = [item for item in path.rglob("*") if item.is_file()]
    return (len(files), sum(item.stat().st_size for item in files))


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
    """Watches cell executions and saves expensive results to Drive automatically.

    Turned on by ``init(autosave=True, ...)``. It registers a callback on IPython's
    ``post_run_cell`` event - the very signal behind the run-count and the success/error
    mark next to each cell - so it runs after every cell. When a cell that trained a model
    finishes, the model is exported to Drive and recorded in the manifest, tagged with the
    cell that produced it. You can also mark things by hand with ``keep(...)`` /
    ``keep_folder(...)``, and have a data folder auto-saved when it grows by registering it
    with ``watch(...)``.
    """

    def __init__(self, lesson: str):
        self.lesson = lesson
        self._registered = False
        self._watched = {}  # name -> [Path, last_signature]
        self._ignored = set()
        self._execution_count = 0

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

    def watch(self, folder, name: str) -> "AutoSaver":
        """Auto-save a data folder whenever it grows after a data-building cell."""
        path = Path(folder)
        self._watched[name] = [path, _folder_signature(path)]
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
        """Run after every cell; save anything expensive it produced. Never raises."""
        try:
            self._execution_count += 1
            if getattr(result, "success", True) is False:
                return  # a cell that errored did not produce a trustworthy artifact
            cell_source = getattr(getattr(result, "info", None), "raw_cell", "") or ""
            for name, learner in detect_models_to_save(cell_source, _user_namespace()):
                # Skip models the user opted out of, and any that cached_model/keep already
                # saved in this same cell (so we never write a duplicate copy).
                if name in self._ignored or id(learner) in _models_saved_this_cell:
                    continue
                save_model(learner, self.lesson, name, self._source(cell_source))
            self._save_grown_folders(cell_source)
        except Exception as error:  # autosave must never break the user's notebook
            _persist_log(f"autosave skipped a cell after an error: {error!r}")
        finally:
            # Reset the per-cell de-dup set so the next cell starts clean.
            _models_saved_this_cell.clear()

    def _save_grown_folders(self, cell_source: str) -> None:
        """Save any watched folder that grew, but only after a data-building cell."""
        if not any(verb in cell_source for verb in DATA_VERBS):
            return
        for name, (path, last_signature) in list(self._watched.items()):
            if name in self._ignored:
                continue
            signature = _folder_signature(path)
            if signature != last_signature and signature[0] > 0:
                save_folder(path, self.lesson, name, self._source(cell_source))
                self._watched[name][1] = signature


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


def _commit_once(repo_root: Path, message_prefix: str, quiet_when_clean: bool = False) -> bool:
    """Stage notebooks + manifests and commit if anything changed. Returns True if so."""
    root = str(repo_root)
    for spec in _snapshot_pathspecs():
        # Add each pathspec on its own. A spec that matches nothing (for example, no
        # manifests have been created yet) makes ``git add`` print a harmless "did not
        # match" and exit non-zero; combining specs into one call would let that abort the
        # whole add, so we keep them separate and swallow that stderr.
        subprocess.run(
            ["git", "-C", root, "add", "--", spec],
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
    return True


def snapshot(repo_root=None, once: bool = True, interval: float = 5.0,
             message_prefix: str = "snapshot") -> bool:
    """Commit notebooks + manifests to git. Returns True if a commit was made.

    Runs only where a git repo exists (your own machine); on Colab there is no repo, so
    this is a logged no-op. With ``once=False`` it polls and commits whenever files
    change, until interrupted - that is what ``bin/snapshot --auto`` runs. Running
    ``--once`` is independent: it commits and exits without touching any ``--auto`` loop.
    """
    root = find_repo_root(repo_root)
    if root is None:
        _persist_log("snapshot skipped: no git repository here (expected on Colab)")
        return False
    if once:
        return _commit_once(root, message_prefix)
    _persist_log("snapshot --auto watching for changes; press Ctrl+C to stop")
    try:
        while True:
            _commit_once(root, message_prefix, quiet_when_clean=True)
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
        if entry.get("kind") == "dataset" and source.suffix == ".zip":
            # Put the images back exactly where the notebook reads them, using the path we
            # recorded at save time. A relative recorded path is resolved against dest_root;
            # an absolute one is used as-is; with no recorded path we fall back generically.
            recorded = entry.get("restore_to")
            if recorded:
                candidate = Path(recorded)
                target_dir = candidate if candidate.is_absolute() else dest_root / candidate
            else:
                target_dir = dest_root / "datasets" / name
            unzip_dir(source, target_dir)
        else:
            subdir = "models" if entry.get("kind") == "model" else "datasets"
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


def _default_lesson() -> str:
    """Tag to use when ``init`` is called with autosave on but no explicit lesson name."""
    name = current_notebook()
    return os.path.splitext(name)[0] if name else "lesson"


def init(
    packages: Sequence[str] = ("fastai",),
    setup_book: bool = False,
    competition: Optional[str] = None,
    mount_drive: bool = True,
    wide_print: bool = False,
    internet_check: bool = False,
    autosave: bool = True,
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

    return SetupContext(
        device=device,
        in_colab=env.in_colab,
        iskaggle=env.iskaggle,
        iscolab=env.iscolab,
        path=path,
        saver=saver,
        drive_dir=drive_root(),
    )


def _main(argv=None):
    """Command-line entry point used by ``bin/snapshot`` and ``bin/restore``.

    Examples:
        python course_setup.py snapshot --once
        python course_setup.py snapshot --auto
        python course_setup.py restore lesson-1
    """
    import argparse

    parser = argparse.ArgumentParser(description="course_setup persistence commands")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot", help="commit notebooks + manifests to git")
    mode = snap.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true", help="watch and commit continuously")
    mode.add_argument("--once", action="store_true", help="commit once and exit (default)")

    rest = sub.add_parser("restore", help="copy a lesson's artifacts from Drive into the repo")
    rest.add_argument("lesson", help="lesson name, e.g. lesson-1")
    rest.add_argument(
        "--dest",
        default=None,
        help="base directory to restore into (defaults to the current directory)",
    )

    args = parser.parse_args(argv)
    if args.command == "snapshot":
        snapshot(once=not args.auto)
    elif args.command == "restore":
        restore(args.lesson, dest_root=args.dest)


if __name__ == "__main__":
    _main()
