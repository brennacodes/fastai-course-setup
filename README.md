# fastai-course-setup

A tiny setup helper for my fast.ai course notebooks. It does the boilerplate every notebook
needs - pick the best GPU, detect Colab/Kaggle/local, install the lesson's packages, and (on
Colab) mount Google Drive - so the notebooks do not each repeat it.

The core setup imports only the standard library. The optional notebook round-trip (keeping
notebook files in sync between Colab and your Mac, with merges) uses `nbformat` and `nbdime`;
both are imported lazily, so the module still loads fine where they are absent.

## How notebooks use it

Each notebook starts with a small bootstrap cell that downloads this module the first
time and caches it, then runs setup:

```python
# --- Bootstrap: load the shared course setup module, downloading it once if needed. ---
# Works the same on your Mac, Colab, and Kaggle. The module's source of truth is the
# public repo github.com/brennacodes/fastai-course-setup. We look for a local copy first
# (so offline use and local edits work), then download and cache it under your home dir.
import importlib.util, os, sys, urllib.request

setup_module_url = "https://raw.githubusercontent.com/brennacodes/fastai-course-setup/main/course_setup.py"
setup_cache_dir = os.path.expanduser("~/.fastai-course-setup")

if importlib.util.find_spec("course_setup") is None:
    search_dirs = [os.getcwd(), *sys.path, setup_cache_dir]
    found_dir = next(
        (d for d in search_dirs if d and os.path.exists(os.path.join(d, "course_setup.py"))),
        None,
    )
    if found_dir is None:
        os.makedirs(setup_cache_dir, exist_ok=True)
        try:
            urllib.request.urlretrieve(setup_module_url, os.path.join(setup_cache_dir, "course_setup.py"))
        except Exception as download_error:
            raise RuntimeError(
                f"Could not download course_setup.py from {setup_module_url} "
                f"({download_error}). Put a copy in {setup_cache_dir} or next to this "
                "notebook, then re-run this cell."
            ) from download_error
        found_dir = setup_cache_dir
    if found_dir not in sys.path:
        sys.path.insert(0, found_dir)

import course_setup
```

```python
context = course_setup.init(packages=("fastai",))
device = context.device
```

`init` returns a `context` with `device`, `in_colab`, `iskaggle`, `iscolab`, and `path`
(the downloaded competition folder, when one was requested).

The notebook still keeps its own `from fastai.vision.all import *` (or
`from fastbook import *`) line. Python's `import *` only adds names to the namespace where
it runs, so the star-import has to live in the notebook; this module only guarantees the
package is installed.

## API

- `init(packages=("fastai",), setup_book=False, competition=None, mount_drive=True, wide_print=False, internet_check=False, autosave=True, auto_sync=True, lesson=None)`
- `detect_env()`, `ensure_packages(packages)`, `mount_colab_drive()`, `select_device()`,
  `download_competition(name)`, `set_wide_print()`, `check_internet()`

When `auto_sync` is on (the default), `init` keeps the notebook *file* flowing too: on a
machine with a git repo (your Mac) a background daemon pulls Colab edits home, mirrors
notebooks to Drive, and commits + pushes on a timer; on Colab it exports the live notebook to
Drive so the Mac daemon can complete the round-trip. A per-repo lock keeps a terminal
`bin/sync` from double-committing alongside the in-kernel daemon.

## Command-line persistence helpers

```
python course_setup.py snapshot --auto   # commit + push notebooks as they change
python course_setup.py mirror --auto     # copy notebooks out to Drive
python course_setup.py pull --auto       # bring Colab edits back, merging with nbdime
python course_setup.py sync --auto       # pull + mirror + snapshot in one loop
python course_setup.py resolve <path>    # accept a hand-resolved notebook after a conflict
python course_setup.py restore <lesson>  # restore a lesson's artifacts from Drive
```

The `pull`/`sync`/`resolve` notebook round-trip needs `nbformat` and `nbdime` installed
(`pip install nbdime`).

## Tests

```
python -m unittest test_course_setup
```

The core setup tests use the standard library only - no torch or fastai needed. The notebook
round-trip tests (pull, merge, export) additionally need `nbformat` and `nbdime` installed.
