"""Unit tests for course_setup.

These use only the standard library (``unittest`` plus ``unittest.mock``), so they run
with ``python -m unittest`` or ``python -m pytest`` without installing torch, fastai, or
any other heavy dependency. Everything those libraries would do is replaced with a mock,
which lets us assert on the decisions the module makes (which device, install or skip,
which path) rather than on real hardware or network.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import course_setup


@contextmanager
def temp_drive():
    """Point course_setup's Drive at a throwaway directory for the duration of a test.

    Setting FASTAI_DRIVE_ROOT makes ``drive_root()`` resolve there, so the persistence
    helpers read and write into an isolated temp folder instead of a real Google Drive.
    """
    with tempfile.TemporaryDirectory() as directory:
        with mock.patch.dict("os.environ", {"FASTAI_DRIVE_ROOT": directory}):
            yield Path(directory)


class FakeLearner:
    """Stand-in for a fastai Learner: exports a tiny file and can 'predict'.

    ``_looks_like_learner`` only checks for callable ``export`` and ``predict``, so this
    is enough to exercise the model-saving paths without importing fastai or torch.
    """

    def __init__(self, payload=b"fake-model-bytes"):
        self.payload = payload
        self.export_calls = []

    def export(self, path):
        self.export_calls.append(Path(path))
        Path(path).write_bytes(self.payload)

    def predict(self, item):  # pragma: no cover - presence is what matters
        return "pill"


class FakeFrame:
    """Stand-in for a pandas DataFrame: has to_csv and columns, so _saveable_kind sees it
    as a 'dataframe' (and not as a learner, since it has no export/predict)."""

    columns = ["a"]

    def to_csv(self, path, index=False):
        Path(path).write_text("a\n")


class DetectEnvTests(unittest.TestCase):
    def test_local_when_no_platform_markers(self):
        # google.colab set to None makes "import google.colab" raise -> not Colab;
        # an empty environment means not Kaggle and not Colab.
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.dict(
            sys.modules, {"google.colab": None}
        ):
            env = course_setup.detect_env()
        self.assertFalse(env.in_colab)
        self.assertFalse(env.iskaggle)
        self.assertFalse(env.iscolab)

    def test_kaggle_detected_from_env_var(self):
        with mock.patch.dict(
            "os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=True
        ):
            with mock.patch.dict(sys.modules, {"google.colab": None}):
                env = course_setup.detect_env()
        self.assertTrue(env.iskaggle)
        self.assertFalse(env.iscolab)

    def test_colab_detected_from_env_var(self):
        with mock.patch.dict("os.environ", {"COLAB_GPU": "1"}, clear=True):
            with mock.patch.dict(sys.modules, {"google.colab": None}):
                env = course_setup.detect_env()
        self.assertTrue(env.iscolab)


class EnsurePackagesTests(unittest.TestCase):
    def test_installs_only_missing_packages(self):
        # "fastai" is present (find_spec returns an object); "timm" is missing (None).
        def fake_find_spec(name):
            return object() if name == "fastai" else None

        with mock.patch.object(
            course_setup.importlib.util, "find_spec", side_effect=fake_find_spec
        ), mock.patch.object(subprocess, "run") as fake_run:
            course_setup.ensure_packages(["fastai", "timm"])

        fake_run.assert_called_once()
        called_args = fake_run.call_args.args[0]
        # Installs the missing one, via subprocess as a list (never a shell string),
        # using the current interpreter's pip.
        self.assertIsInstance(called_args, list)
        self.assertEqual(called_args[:3], [sys.executable, "-m", "pip"])
        self.assertIn("timm", called_args)

    def test_skips_when_all_present(self):
        with mock.patch.object(
            course_setup.importlib.util, "find_spec", return_value=object()
        ), mock.patch.object(subprocess, "run") as fake_run:
            course_setup.ensure_packages(["fastai"])
        fake_run.assert_not_called()

    def test_version_pinned_spec_probes_module_name_only(self):
        captured = {}

        def fake_find_spec(name):
            captured["probe"] = name
            return None

        with mock.patch.object(
            course_setup.importlib.util, "find_spec", side_effect=fake_find_spec
        ), mock.patch.object(subprocess, "run") as fake_run:
            course_setup.ensure_packages(["dtreeviz==1.4.1"])

        # The import probe strips the version, but the full spec is what gets installed.
        self.assertEqual(captured["probe"], "dtreeviz")
        self.assertIn("dtreeviz==1.4.1", fake_run.call_args.args[0])


class SelectDeviceTests(unittest.TestCase):
    def _fake_torch(self, cuda=False, mps=False):
        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = cuda
        fake_torch.backends.mps.is_available.return_value = mps
        fake_torch.get_device_name.return_value = "Tesla T4"
        # torch.device(name) just records the name we asked for.
        fake_torch.device.side_effect = lambda name: f"device:{name}"
        return fake_torch

    def _run_with_torch(self, fake_torch):
        # fastai import is allowed to fail (covered by the try/except in select_device).
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.dict(
            sys.modules, {"fastai.torch_core": None}
        ):
            return course_setup.select_device()

    def test_prefers_cuda(self):
        device = self._run_with_torch(self._fake_torch(cuda=True, mps=True))
        self.assertEqual(device, "device:cuda")

    def test_falls_back_to_mps(self):
        device = self._run_with_torch(self._fake_torch(cuda=False, mps=True))
        self.assertEqual(device, "device:mps")

    def test_falls_back_to_cpu(self):
        device = self._run_with_torch(self._fake_torch(cuda=False, mps=False))
        self.assertEqual(device, "device:cpu")


class DownloadCompetitionTests(unittest.TestCase):
    def test_kaggle_uses_input_mount(self):
        env = course_setup.Environment(in_colab=False, iskaggle=True, iscolab=False)
        path = course_setup.download_competition("titanic", env)
        self.assertEqual(path, Path("../input/titanic"))

    def test_local_returns_existing_path_without_downloading(self):
        env = course_setup.Environment(in_colab=False, iskaggle=False, iscolab=False)
        with mock.patch.object(Path, "exists", return_value=True):
            path = course_setup.download_competition("titanic", env)
        self.assertEqual(path, Path("titanic"))

    def test_local_downloads_and_unzips_when_missing(self):
        env = course_setup.Environment(in_colab=False, iskaggle=False, iscolab=False)
        fake_kaggle = mock.MagicMock()
        with mock.patch.object(Path, "exists", return_value=False), mock.patch.dict(
            sys.modules, {"kaggle": fake_kaggle}
        ), mock.patch.object(course_setup.zipfile, "ZipFile") as fake_zip:
            path = course_setup.download_competition("titanic", env)
        fake_kaggle.api.competition_download_cli.assert_called_once_with("titanic")
        fake_zip.assert_called_once_with("titanic.zip")
        self.assertEqual(path, Path("titanic"))


class CheckInternetTests(unittest.TestCase):
    def setUp(self):
        # check_internet sets a process-wide default socket timeout; restore it after.
        import socket

        previous = socket.getdefaulttimeout()
        self.addCleanup(socket.setdefaulttimeout, previous)

    def test_raises_clear_error_when_offline(self):
        failing_socket = mock.MagicMock()
        failing_socket.connect.side_effect = OSError("unreachable")
        with mock.patch.object(course_setup.socket, "socket", return_value=failing_socket):
            with self.assertRaises(RuntimeError) as caught:
                course_setup.check_internet()
        self.assertIn("No internet", str(caught.exception))

    def test_passes_when_connectable(self):
        with mock.patch.object(course_setup.socket, "socket"):
            course_setup.check_internet()  # should not raise


class MountColabDriveTests(unittest.TestCase):
    def test_noop_off_colab(self):
        env = course_setup.Environment(in_colab=False, iskaggle=False, iscolab=False)
        self.assertFalse(course_setup.mount_colab_drive(env))

    def test_mounts_on_colab(self):
        env = course_setup.Environment(in_colab=True, iskaggle=False, iscolab=True)
        fake_colab = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"google.colab": fake_colab}):
            result = course_setup.mount_colab_drive(env)
        self.assertTrue(result)
        fake_colab.drive.mount.assert_called_once_with("/content/drive")


class InitTests(unittest.TestCase):
    def test_orchestrates_steps_and_returns_context(self):
        env = course_setup.Environment(in_colab=False, iskaggle=False, iscolab=False)
        # A single parent mock records the call order across all the sub-steps.
        manager = mock.Mock()
        with mock.patch.object(course_setup, "detect_env", return_value=env), \
            mock.patch.object(course_setup, "check_internet", manager.check_internet), \
            mock.patch.object(course_setup, "mount_colab_drive", manager.mount_colab_drive), \
            mock.patch.object(course_setup, "ensure_packages", manager.ensure_packages), \
            mock.patch.object(course_setup, "download_competition", manager.download_competition), \
            mock.patch.object(course_setup, "set_wide_print", manager.set_wide_print), \
            mock.patch.object(course_setup, "select_device", manager.select_device), \
            mock.patch.object(course_setup, "_start_auto_sync"):
            manager.download_competition.return_value = Path("titanic")
            manager.select_device.return_value = "device:cpu"
            context = course_setup.init(
                packages=("fastai",), competition="titanic", wide_print=True
            )

        manager.ensure_packages.assert_called_once_with(("fastai",))
        manager.download_competition.assert_called_once_with("titanic", env)
        manager.set_wide_print.assert_called_once()
        manager.mount_colab_drive.assert_called_once()  # mount_drive defaults to True
        manager.check_internet.assert_not_called()  # internet_check defaults to False
        self.assertEqual(context.device, "device:cpu")
        self.assertEqual(context.path, Path("titanic"))
        self.assertFalse(context.iskaggle)

        # Ordering invariant: packages are ensured before the device is chosen, because
        # selecting the device imports torch (which the install brings in).
        call_names = [call[0] for call in manager.mock_calls]
        self.assertLess(
            call_names.index("ensure_packages"), call_names.index("select_device")
        )

    def test_skips_optional_steps_by_default(self):
        env = course_setup.Environment(in_colab=False, iskaggle=False, iscolab=False)
        with mock.patch.object(course_setup, "detect_env", return_value=env), \
            mock.patch.object(course_setup, "mount_colab_drive"), \
            mock.patch.object(course_setup, "ensure_packages"), \
            mock.patch.object(course_setup, "download_competition") as fake_download, \
            mock.patch.object(course_setup, "set_wide_print") as fake_wide, \
            mock.patch.object(course_setup, "select_device", return_value="device:cpu"), \
            mock.patch.object(course_setup, "_start_auto_sync"):
            context = course_setup.init()
        fake_download.assert_not_called()
        fake_wide.assert_not_called()
        self.assertIsNone(context.path)


class SetWidePrintTests(unittest.TestCase):
    def test_calls_each_library_setter(self):
        fake_numpy = mock.MagicMock()
        fake_pandas = mock.MagicMock()
        fake_torch = mock.MagicMock()
        with mock.patch.dict(
            sys.modules,
            {"numpy": fake_numpy, "pandas": fake_pandas, "torch": fake_torch},
        ):
            course_setup.set_wide_print()
        fake_numpy.set_printoptions.assert_called_once()
        fake_torch.set_printoptions.assert_called_once()
        fake_pandas.set_option.assert_called_once()


class DriveLayoutTests(unittest.TestCase):
    def test_drive_root_none_without_drive(self):
        # No override and no detected mount -> None (callers no-op). We patch detection so
        # the result does not depend on whether this machine happens to have Drive mounted.
        with mock.patch.dict("os.environ", {}, clear=True), \
            mock.patch.object(course_setup, "_detect_drive_mount", return_value=None):
            self.assertIsNone(course_setup.drive_root())

    def test_drive_root_uses_detected_mount(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
            mock.patch.object(course_setup, "_detect_drive_mount", return_value=Path("/some/drive")):
            self.assertEqual(course_setup.drive_root(), Path("/some/drive/fastai"))

    def test_override_env_var_wins(self):
        with mock.patch.dict("os.environ", {"FASTAI_DRIVE_ROOT": "/tmp/somewhere"}):
            self.assertEqual(course_setup.drive_root(), Path("/tmp/somewhere"))

    def test_artifacts_dir_created_under_drive(self):
        with temp_drive() as drive:
            directory = course_setup.artifacts_dir("lesson-1")
            self.assertEqual(directory, drive / "artifacts" / "lesson-1")
            self.assertTrue(directory.exists())


class ManifestTests(unittest.TestCase):
    def test_record_and_load_round_trip(self):
        with temp_drive():
            course_setup.record_artifact("lesson-1", "model", {"kind": "model"})
            course_setup.record_artifact("lesson-1", "images", {"kind": "dataset"})
            manifest = course_setup.load_manifest("lesson-1")
        self.assertEqual(manifest["lesson"], "lesson-1")
        self.assertEqual(set(manifest["artifacts"]), {"model", "images"})
        self.assertIn("updated_at", manifest)

    def test_recording_same_name_updates_in_place(self):
        with temp_drive():
            course_setup.record_artifact("lesson-1", "model", {"size_bytes": 1})
            course_setup.record_artifact("lesson-1", "model", {"size_bytes": 2})
            manifest = course_setup.load_manifest("lesson-1")
        self.assertEqual(len(manifest["artifacts"]), 1)
        self.assertEqual(manifest["artifacts"]["model"]["size_bytes"], 2)

    def test_load_manifest_empty_when_absent(self):
        with temp_drive():
            manifest = course_setup.load_manifest("never-saved")
        self.assertEqual(manifest, {"lesson": "never-saved", "artifacts": {}})


class ZipRoundTripTests(unittest.TestCase):
    def test_zip_then_unzip_restores_files(self):
        with tempfile.TemporaryDirectory() as workspace:
            workspace = Path(workspace)
            source = workspace / "data"
            (source / "pill").mkdir(parents=True)
            (source / "pill" / "a.txt").write_text("one")
            (source / "pill" / "b.txt").write_text("two")

            archive = workspace / "data.zip"
            course_setup.zip_dir(source, archive)
            self.assertTrue(archive.exists())

            restored = workspace / "restored"
            course_setup.unzip_dir(archive, restored)
            # Assert inside the block, before the temp directory is cleaned up.
            self.assertEqual((restored / "pill" / "a.txt").read_text(), "one")
            self.assertEqual((restored / "pill" / "b.txt").read_text(), "two")


class SaveModelTests(unittest.TestCase):
    def test_exports_to_drive_and_records_manifest(self):
        with temp_drive() as drive:
            learner = FakeLearner()
            target = course_setup.save_model(learner, "lesson-1", "pill_or_not")
            self.assertEqual(target, drive / "artifacts" / "lesson-1" / "models" / "pill_or_not.pkl")
            self.assertTrue(target.exists())
            self.assertEqual(learner.export_calls, [target])
            manifest = course_setup.load_manifest("lesson-1")
        entry = manifest["artifacts"]["pill_or_not"]
        self.assertEqual(entry["kind"], "model")
        self.assertIn("sha256", entry)
        self.assertEqual(entry["size_bytes"], len(b"fake-model-bytes"))

    def test_noop_without_drive(self):
        # Patch detection so this is a true "no Drive anywhere" case and never touches a
        # real Drive that happens to be mounted on the machine running the tests.
        with mock.patch.dict("os.environ", {}, clear=True), \
            mock.patch.object(course_setup, "_detect_drive_mount", return_value=None):
            self.assertIsNone(course_setup.save_model(FakeLearner(), "lesson-1", "m"))


class CachedHelpersTests(unittest.TestCase):
    def test_cached_model_builds_then_loads_without_rebuilding(self):
        with temp_drive():
            built = []

            def build():
                built.append(True)
                return FakeLearner()

            # Load just returns a marker so we can tell a cache hit from a rebuild.
            loaded = course_setup.cached_model(
                "lesson-1", "pill", build, load_fn=lambda path: ("loaded", path)
            )
            self.assertIsInstance(loaded, FakeLearner)  # first call: a miss, so it built
            self.assertEqual(len(built), 1)

            again = course_setup.cached_model(
                "lesson-1", "pill", build, load_fn=lambda path: ("loaded", path)
            )
        self.assertEqual(again[0], "loaded")  # second call: a hit, loaded not rebuilt
        self.assertEqual(len(built), 1)  # build_fn did not run a second time

    def test_cached_folder_builds_then_restores(self):
        with temp_drive(), tempfile.TemporaryDirectory() as workspace:
            dataset = Path(workspace) / "pill_or_not"

            def build():
                (dataset / "pill").mkdir(parents=True, exist_ok=True)
                (dataset / "pill" / "img.txt").write_text("x")

            course_setup.cached_folder("lesson-1", "pill_or_not", build, dataset)
            self.assertTrue((dataset / "pill" / "img.txt").exists())

            # Wipe the folder and rebuild from the cache; build must not run again.
            rebuilt = []
            for item in dataset.rglob("*"):
                if item.is_file():
                    item.unlink()
            course_setup.cached_folder(
                "lesson-1", "pill_or_not", lambda: rebuilt.append(True), dataset
            )
            self.assertTrue((dataset / "pill" / "img.txt").exists())
            self.assertEqual(rebuilt, [])


class SaveableKindTests(unittest.TestCase):
    def test_learner_is_a_model(self):
        self.assertEqual(course_setup._saveable_kind(FakeLearner()), "model")

    def test_dataframe_is_a_dataframe(self):
        self.assertEqual(course_setup._saveable_kind(FakeFrame()), "dataframe")

    def test_plain_values_are_skipped(self):
        for value in (42, "hi", [1, 2, 3], {"a": 1}):
            self.assertIsNone(course_setup._saveable_kind(value))


class AmbientObjectSaveTests(unittest.TestCase):
    """The hook saves models on training cells and dataframes when they appear."""

    def setUp(self):
        course_setup._models_saved_this_cell.clear()
        self.addCleanup(course_setup._models_saved_this_cell.clear)

    def test_model_saved_only_on_a_training_cell(self):
        saver = course_setup.AutoSaver("lesson-1")
        learner = FakeLearner()
        ns = {"learn": learner}
        with mock.patch.object(course_setup, "_user_namespace", return_value=ns), \
            mock.patch.object(course_setup, "save_model") as fake_save:
            saver._save_namespace_objects("learn = vision_learner(dls, resnet18)")
        fake_save.assert_not_called()  # constructed but not trained -> not saved

        with mock.patch.object(course_setup, "_user_namespace", return_value=ns), \
            mock.patch.object(course_setup, "save_model") as fake_save:
            saver._save_namespace_objects("learn.fine_tune(3)")
        fake_save.assert_called_once()  # trained -> saved

    def test_new_dataframe_is_saved(self):
        saver = course_setup.AutoSaver("lesson-1")
        with mock.patch.object(course_setup, "_user_namespace", return_value={"df": FakeFrame()}), \
            mock.patch.object(course_setup, "save_dataframe") as fake_save:
            saver._save_namespace_objects("df = pd.read_csv(url)")
        fake_save.assert_called_once()

    def test_underscored_and_ignored_names_are_skipped(self):
        saver = course_setup.AutoSaver("lesson-1").ignore("learn")
        ns = {"_hidden": FakeLearner(), "learn": FakeLearner()}
        with mock.patch.object(course_setup, "_user_namespace", return_value=ns), \
            mock.patch.object(course_setup, "save_model") as fake_save:
            saver._save_namespace_objects("learn.fine_tune(1)")
        fake_save.assert_not_called()


class AmbientFileSaveTests(unittest.TestCase):
    """The hook mirrors files/folders a cell writes under the working directory."""

    def test_first_cell_only_sets_a_baseline(self):
        saver = course_setup.AutoSaver("lesson-1")
        with mock.patch.object(course_setup, "save_file") as ffile, \
            mock.patch.object(course_setup, "save_folder") as ffolder:
            saver._save_new_files("")  # _file_index is None -> just record the baseline
        ffile.assert_not_called()
        ffolder.assert_not_called()
        self.assertIsNotNone(saver._file_index)

    def test_loose_file_is_saved_individually(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                saver = course_setup.AutoSaver("lesson-1")
                saver._file_index = {}  # baseline already taken, nothing existed
                Path("result.pkl").write_text("x")
                with mock.patch.object(course_setup, "save_file") as ffile, \
                    mock.patch.object(course_setup, "save_folder") as ffolder:
                    saver._save_new_files("learn.export('result.pkl')")
                ffile.assert_called_once()
                self.assertEqual(Path(ffile.call_args.args[0]).name, "result.pkl")
                ffolder.assert_not_called()
            finally:
                os.chdir(original_cwd)

    def test_changed_subdirectory_is_zipped_once(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                saver = course_setup.AutoSaver("lesson-1")
                saver._file_index = {}
                (Path("data") / "imgs").mkdir(parents=True)
                (Path("data") / "imgs" / "a.jpg").write_text("x")
                (Path("data") / "imgs" / "b.jpg").write_text("y")
                with mock.patch.object(course_setup, "save_file") as ffile, \
                    mock.patch.object(course_setup, "save_folder") as ffolder:
                    saver._save_new_files("download_images(...)")
                ffolder.assert_called_once()
                self.assertEqual(Path(ffolder.call_args.args[0]).name, "data")
                ffile.assert_not_called()
            finally:
                os.chdir(original_cwd)


class KeepTests(unittest.TestCase):
    def test_keep_saves_a_learner_deterministically(self):
        with temp_drive() as drive:
            saver = course_setup.AutoSaver("lesson-1")
            first = saver.keep(FakeLearner(), "explicit")
            second = saver.keep(FakeLearner(), "explicit")
            expected = drive / "artifacts" / "lesson-1" / "models" / "explicit.pkl"
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)


class SnapshotTests(unittest.TestCase):
    @contextmanager
    def _temp_git_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            subprocess.run(["git", "-C", directory, "config", "user.email", "t@t.test"], check=True)
            subprocess.run(["git", "-C", directory, "config", "user.name", "Test"], check=True)
            yield root

    def _commit_count(self, root):
        result = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--count", "HEAD"],
            capture_output=True, text=True,
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0

    def test_commits_a_changed_notebook(self):
        with self._temp_git_repo() as root:
            (root / "demo.ipynb").write_text("{}")
            made = course_setup.snapshot(repo_root=root, once=True)
            self.assertTrue(made)
            self.assertEqual(self._commit_count(root), 1)

    def test_nothing_to_commit_returns_false(self):
        with self._temp_git_repo() as root:
            (root / "demo.ipynb").write_text("{}")
            course_setup.snapshot(repo_root=root, once=True)
            again = course_setup.snapshot(repo_root=root, once=True)
            self.assertFalse(again)
            self.assertEqual(self._commit_count(root), 1)

    def test_no_repo_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(course_setup.snapshot(repo_root=directory, once=True))

    @contextmanager
    def _temp_git_repo_with_remote(self):
        """A working repo wired to a bare 'origin' with an upstream already set.

        ``_push_to_remote`` runs a bare ``git push`` (no args), which needs the branch to have
        an upstream. So we seed one commit and push it with ``-u`` to establish that upstream;
        tests then assert that a later snapshot's push moves the remote forward.
        """
        with tempfile.TemporaryDirectory() as work_dir, tempfile.TemporaryDirectory() as remote_dir:
            root = Path(work_dir)
            subprocess.run(["git", "-C", work_dir, "init", "-q"], check=True)
            subprocess.run(["git", "-C", work_dir, "config", "user.email", "t@t.test"], check=True)
            subprocess.run(["git", "-C", work_dir, "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "init", "--bare", "-q", remote_dir], check=True)
            subprocess.run(["git", "-C", work_dir, "remote", "add", "origin", remote_dir], check=True)
            (root / "seed.txt").write_text("seed")
            subprocess.run(["git", "-C", work_dir, "add", "seed.txt"], check=True)
            subprocess.run(["git", "-C", work_dir, "commit", "-q", "-m", "seed"], check=True)
            subprocess.run(["git", "-C", work_dir, "push", "-q", "-u", "origin", "HEAD"], check=True)
            yield root, Path(remote_dir)

    def _head(self, root):
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    def test_pushes_after_commit(self):
        with self._temp_git_repo_with_remote() as (root, remote):
            (root / "demo.ipynb").write_text("{}")
            self.assertTrue(course_setup.snapshot(repo_root=root, once=True))
            # The new commit reached the bare remote: both point at the same HEAD.
            self.assertEqual(self._head(root), self._head(remote))

    def test_push_false_commits_without_pushing(self):
        with self._temp_git_repo_with_remote() as (root, remote):
            remote_before = self._head(remote)
            (root / "demo.ipynb").write_text("{}")
            self.assertTrue(course_setup.snapshot(repo_root=root, once=True, push=False))
            # Local advanced, remote did not.
            self.assertNotEqual(self._head(root), self._head(remote))
            self.assertEqual(self._head(remote), remote_before)

    def test_push_failure_is_swallowed(self):
        # A repo with no remote: the push fails, but the commit must still succeed and the
        # call must not raise.
        with self._temp_git_repo() as root:
            (root / "demo.ipynb").write_text("{}")
            self.assertTrue(course_setup.snapshot(repo_root=root, once=True))
            self.assertEqual(self._commit_count(root), 1)

    def test_push_is_attempted_on_every_commit(self):
        # The commit path must actually call the push, not just optionally; a real failing push
        # is covered above, here we assert the attempt happens and does not change the result.
        with self._temp_git_repo() as root:
            (root / "demo.ipynb").write_text("{}")
            with mock.patch.object(course_setup, "_push_to_remote", return_value=False) as push:
                self.assertTrue(course_setup.snapshot(repo_root=root, once=True))
                push.assert_called_once()


class MirrorTests(unittest.TestCase):
    """The Mac-side notebook-to-Drive mirror (mirror_notebooks)."""

    @contextmanager
    def _temp_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            yield Path(directory)

    def test_copies_a_notebook_into_drive(self):
        with self._temp_repo() as root, temp_drive() as drive:
            (root / "demo.ipynb").write_text('{"cells": []}')
            copied = course_setup.mirror_notebooks(repo_root=root, once=True)
            self.assertEqual(copied, 1)
            mirrored = drive / "notebooks" / "demo.ipynb"
            self.assertTrue(mirrored.exists())
            self.assertEqual(mirrored.read_text(), '{"cells": []}')

    def test_preserves_subdirectory_structure(self):
        with self._temp_repo() as root, temp_drive() as drive:
            nested = root / "homework" / "lesson-1"
            nested.mkdir(parents=True)
            (nested / "work.ipynb").write_text("{}")
            course_setup.mirror_notebooks(repo_root=root, once=True)
            self.assertTrue(
                (drive / "notebooks" / "homework" / "lesson-1" / "work.ipynb").exists()
            )

    def test_non_notebook_files_are_ignored(self):
        with self._temp_repo() as root, temp_drive() as drive:
            (root / "demo.ipynb").write_text("{}")
            (root / "notes.txt").write_text("hello")
            (root / "demo.ipynb.bak").write_text("{}")
            copied = course_setup.mirror_notebooks(repo_root=root, once=True)
            self.assertEqual(copied, 1)
            self.assertFalse((drive / "notebooks" / "notes.txt").exists())
            self.assertFalse((drive / "notebooks" / "demo.ipynb.bak").exists())

    def test_unchanged_notebook_is_not_recopied(self):
        with self._temp_repo() as root, temp_drive():
            (root / "demo.ipynb").write_text("{}")
            self.assertEqual(course_setup.mirror_notebooks(repo_root=root, once=True), 1)
            # A second pass with no edits copies nothing (content matches the Drive copy).
            self.assertEqual(course_setup.mirror_notebooks(repo_root=root, once=True), 0)

    def test_edited_notebook_is_recopied(self):
        with self._temp_repo() as root, temp_drive():
            notebook = root / "demo.ipynb"
            notebook.write_text("{}")
            course_setup.mirror_notebooks(repo_root=root, once=True)
            notebook.write_text('{"cells": [1]}')
            self.assertEqual(course_setup.mirror_notebooks(repo_root=root, once=True), 1)

    def test_no_repo_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory, temp_drive():
            self.assertEqual(
                course_setup.mirror_notebooks(repo_root=directory, once=True), 0
            )

    def test_no_drive_is_a_noop(self):
        with self._temp_repo() as root:
            (root / "demo.ipynb").write_text("{}")
            with mock.patch.object(course_setup, "drive_root", return_value=None):
                self.assertEqual(
                    course_setup.mirror_notebooks(repo_root=root, once=True), 0
                )

    def test_auto_mtime_gate_skips_unchanged_files(self):
        # The --auto loop passes a source_mtimes dict; a file whose mtime has not changed
        # since the last pass is skipped entirely (not even re-checksummed or copied).
        with self._temp_repo() as root, temp_drive() as drive:
            (root / "demo.ipynb").write_text("{}")
            dest_base = drive / "notebooks"
            source_mtimes = {}
            self.assertEqual(
                course_setup._mirror_once(root, dest_base, source_mtimes), 1
            )
            self.assertEqual(
                course_setup._mirror_once(root, dest_base, source_mtimes), 0
            )

    def test_noise_directories_are_pruned(self):
        # Notebooks buried in version-control/cache/venv directories are never mirrored.
        with self._temp_repo() as root, temp_drive() as drive:
            for noisy_dir in (".git", "__pycache__", ".venv"):
                buried = root / noisy_dir
                buried.mkdir(parents=True, exist_ok=True)
                (buried / "buried.ipynb").write_text("{}")
            (root / "real.ipynb").write_text("{}")
            copied = course_setup.mirror_notebooks(repo_root=root, once=True)
            self.assertEqual(copied, 1)
            self.assertTrue((drive / "notebooks" / "real.ipynb").exists())
            self.assertFalse((drive / "notebooks" / ".git" / "buried.ipynb").exists())


class PullTests(unittest.TestCase):
    """The Mac-side round-trip that brings Colab edits home (pull_notebooks + merge).

    These build real notebooks with nbformat so the three sides share stable cell ids, which
    is what lets nbdime tell a genuine conflict from two edits to different cells.
    """

    @contextmanager
    def _temp_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            yield Path(directory)

    def _make_notebook(self, path, sources):
        import nbformat
        from nbformat.v4 import new_markdown_cell, new_notebook

        node = new_notebook(cells=[new_markdown_cell(text) for text in sources])
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(node, str(path))

    def _edit_cell(self, path, index, new_source):
        import nbformat

        node = nbformat.read(str(path), as_version=4)
        node.cells[index].source = new_source
        nbformat.write(node, str(path))

    def _cell_sources(self, path):
        import nbformat

        node = nbformat.read(str(path), as_version=4)
        return [cell.source for cell in node.cells]

    def test_new_local_notebook_seeds_drive_and_base(self):
        with self._temp_repo() as root, temp_drive() as drive:
            self._make_notebook(root / "demo.ipynb", ["alpha", "beta"])
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 0)
            self.assertTrue((drive / "notebooks" / "demo.ipynb").exists())
            self.assertTrue((drive / "notebooks" / ".sync-base" / "demo.ipynb").exists())

    def test_new_colab_notebook_is_pulled_down(self):
        with self._temp_repo() as root, temp_drive() as drive:
            self._make_notebook(drive / "notebooks" / "demo.ipynb", ["from colab"])
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 1)
            self.assertEqual(self._cell_sources(root / "demo.ipynb"), ["from colab"])

    def test_only_drive_changed_updates_mac(self):
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            self._edit_cell(drive / "notebooks" / "demo.ipynb", 1, "beta on colab")
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 1)
            self.assertEqual(self._cell_sources(mac)[1], "beta on colab")

    def test_only_mac_changed_updates_drive(self):
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            self._edit_cell(mac, 0, "alpha on mac")
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 0)
            self.assertEqual(
                self._cell_sources(drive / "notebooks" / "demo.ipynb")[0], "alpha on mac"
            )

    def test_clean_merge_of_disjoint_edits(self):
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            self._edit_cell(mac, 0, "alpha on mac")
            self._edit_cell(drive / "notebooks" / "demo.ipynb", 1, "beta on colab")
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 1)
            self.assertEqual(self._cell_sources(mac), ["alpha on mac", "beta on colab"])
            self.assertFalse((root / "demo.merge-conflict.ipynb").exists())

    def test_disjoint_line_edits_in_one_cell_merge_cleanly(self):
        # nbdime merges a cell's source line by line, so edits to different lines of the SAME
        # cell auto-merge without a conflict and keep both. Verify that, so we know this common
        # case is not silently lossy.
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["line one\nline two\nline three"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            self._edit_cell(mac, 0, "LINE ONE (mac)\nline two\nline three")
            self._edit_cell(
                drive / "notebooks" / "demo.ipynb", 0, "line one\nline two\nLINE THREE (colab)"
            )
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 1)
            merged = self._cell_sources(mac)[0]
            self.assertIn("LINE ONE (mac)", merged)
            self.assertIn("LINE THREE (colab)", merged)
            self.assertFalse((root / "demo.merge-conflict.ipynb").exists())

    def test_conflicting_edits_pause_without_overwriting(self):
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            base_file = drive / "notebooks" / ".sync-base" / "demo.ipynb"
            base_before = base_file.read_text()
            self._edit_cell(mac, 0, "alpha MAC")
            self._edit_cell(drive / "notebooks" / "demo.ipynb", 0, "alpha COLAB")
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 0)
            # The Mac copy is untouched, a conflict scratch is written, and the base is intact.
            self.assertEqual(self._cell_sources(mac)[0], "alpha MAC")
            self.assertTrue((root / "demo.merge-conflict.ipynb").exists())
            self.assertEqual(base_file.read_text(), base_before)

    def test_resolve_conflict_advances_base_and_clears_scratch(self):
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            self._edit_cell(mac, 0, "alpha MAC")
            self._edit_cell(drive / "notebooks" / "demo.ipynb", 0, "alpha COLAB")
            course_setup.pull_notebooks(repo_root=root, once=True)  # produces the conflict
            # The user resolves by editing the real notebook, then accepts it.
            self._edit_cell(mac, 0, "alpha RESOLVED")
            self.assertTrue(course_setup.resolve_conflict("demo.ipynb", repo_root=root))
            self.assertFalse((root / "demo.merge-conflict.ipynb").exists())
            self.assertEqual(
                self._cell_sources(drive / "notebooks" / "demo.ipynb")[0], "alpha RESOLVED"
            )
            # Everything now agrees, so a further pass has nothing to do.
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 0)

    def test_conflict_scratch_is_not_committed(self):
        with self._temp_repo() as root, temp_drive() as drive:
            subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t.test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base
            self._edit_cell(mac, 0, "alpha MAC")
            self._edit_cell(drive / "notebooks" / "demo.ipynb", 0, "alpha COLAB")
            course_setup.pull_notebooks(repo_root=root, once=True)  # produces the conflict
            self.assertTrue((root / "demo.merge-conflict.ipynb").exists())
            course_setup.snapshot(repo_root=root, once=True, push=False)
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files"], capture_output=True, text=True
            ).stdout
            self.assertIn("demo.ipynb", tracked)
            self.assertNotIn("merge-conflict", tracked)

    def test_unreadable_mac_copy_is_not_overwritten(self):
        # If the Mac file exists but is momentarily unparseable (mid-save), pull must not treat
        # it as absent and copy Drive down over it. It should skip and leave the Mac bytes alone.
        with self._temp_repo() as root, temp_drive() as drive:
            mac = root / "demo.ipynb"
            self._make_notebook(mac, ["alpha", "beta"])
            course_setup.pull_notebooks(repo_root=root, once=True)  # seed base + Drive
            self._edit_cell(drive / "notebooks" / "demo.ipynb", 0, "newer on colab")
            mac.write_text("{ this is not valid notebook json")  # a half-written save
            self.assertEqual(course_setup.pull_notebooks(repo_root=root, once=True), 0)
            self.assertEqual(mac.read_text(), "{ this is not valid notebook json")


class ExportTests(unittest.TestCase):
    """The Colab-side notebook export (export_notebook_to_drive) and its Drive path index."""

    @contextmanager
    def _temp_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "-C", directory, "init", "-q"], check=True)
            yield Path(directory)

    def _make_notebook(self, path, sources):
        import nbformat
        from nbformat.v4 import new_markdown_cell, new_notebook

        node = new_notebook(cells=[new_markdown_cell(text) for text in sources])
        path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(node, str(path))

    @contextmanager
    def _fake_colab(self, notebook_dict=None, raises=False):
        """Inject a stand-in google.colab._message so export runs as if on Colab."""
        import types

        message = types.ModuleType("google.colab._message")
        message.calls = []

        def blocking_request(*args, **kwargs):
            message.calls.append((args, kwargs))
            if raises:
                raise RuntimeError("colab internals changed")
            return {"ipynb": notebook_dict}

        message.blocking_request = blocking_request
        colab = types.ModuleType("google.colab")
        colab._message = message
        google = types.ModuleType("google")
        google.colab = colab
        with mock.patch.dict(
            sys.modules,
            {"google": google, "google.colab": colab, "google.colab._message": message},
        ):
            yield message

    def test_returns_false_without_drive(self):
        with mock.patch.object(course_setup, "drive_root", return_value=None):
            self.assertFalse(course_setup.export_notebook_to_drive())

    def test_returns_false_off_colab(self):
        with temp_drive():
            # No google.colab in this environment, so the import fails and export bails out.
            self.assertFalse(course_setup.export_notebook_to_drive())

    def test_failure_is_loud_but_harmless(self):
        with temp_drive(), self._fake_colab(raises=True):
            # blocking_request raises (Colab 'changed'); export must swallow it, not crash.
            self.assertFalse(course_setup.export_notebook_to_drive())

    def test_writes_to_resolved_repo_path(self):
        import nbformat

        with self._temp_repo() as root, temp_drive() as drive:
            nested = root / "homework" / "lesson-1" / "work.ipynb"
            self._make_notebook(nested, ["original"])
            course_setup.mirror_notebooks(repo_root=root, once=True)  # writes the path index
            exported = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_markdown_cell("edited on colab")]
            )
            notebook_dict = json.loads(nbformat.writes(exported))
            with self._fake_colab(notebook_dict=notebook_dict) as colab_message, mock.patch.object(
                course_setup, "current_notebook", return_value="work.ipynb"
            ):
                self.assertTrue(course_setup.export_notebook_to_drive())
            # It asked Colab for the notebook via the documented message type.
            self.assertEqual(colab_message.calls[0][0][0], "get_ipynb")
            written = drive / "notebooks" / "homework" / "lesson-1" / "work.ipynb"
            self.assertTrue(written.exists())
            self.assertEqual(
                nbformat.read(str(written), as_version=4).cells[0].source, "edited on colab"
            )

    def test_skips_when_basename_is_ambiguous(self):
        with self._temp_repo() as root, temp_drive() as drive:
            self._make_notebook(root / "a" / "work.ipynb", ["a"])
            self._make_notebook(root / "b" / "work.ipynb", ["b"])
            course_setup.mirror_notebooks(repo_root=root, once=True)  # two 'work.ipynb' in index
            self.assertIsNone(
                course_setup._resolve_export_relpath(drive / "notebooks", "work.ipynb")
            )

    def test_unindexed_notebook_lands_in_from_colab_folder(self):
        # A notebook created on Colab (never mirrored, so not in the index) must not be dropped;
        # it lands under _from-colab/ for the Mac to pick up.
        import nbformat

        with temp_drive() as drive:
            exported = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("born here")])
            notebook_dict = json.loads(nbformat.writes(exported))
            with self._fake_colab(notebook_dict=notebook_dict), mock.patch.object(
                course_setup, "current_notebook", return_value="novel.ipynb"
            ):
                self.assertTrue(course_setup.export_notebook_to_drive())
            landed = drive / "notebooks" / "_from-colab" / "novel.ipynb"
            self.assertTrue(landed.exists())
            self.assertEqual(
                nbformat.read(str(landed), as_version=4).cells[0].source, "born here"
            )


class AutoSyncTests(unittest.TestCase):
    """The init-driven auto-sync: the SyncDaemon's pass and which half _start_auto_sync picks."""

    def _env(self, in_colab=False, iscolab=False):
        return course_setup.Environment(in_colab=in_colab, iskaggle=False, iscolab=iscolab)

    def test_daemon_run_once_calls_pull_mirror_snapshot(self):
        with mock.patch.object(course_setup, "pull_notebooks") as pull, mock.patch.object(
            course_setup, "mirror_notebooks"
        ) as mirror, mock.patch.object(course_setup, "snapshot") as snap:
            course_setup.SyncDaemon("/tmp/repo").run_once()
            pull.assert_called_once_with(repo_root=Path("/tmp/repo"), once=True)
            mirror.assert_called_once_with(repo_root=Path("/tmp/repo"), once=True)
            snap.assert_called_once_with(repo_root=Path("/tmp/repo"), once=True)

    def test_daemon_isolates_a_failing_step(self):
        # A failing pull must not stop mirror and snapshot from running in the same pass.
        with mock.patch.object(
            course_setup, "pull_notebooks", side_effect=RuntimeError("boom")
        ), mock.patch.object(course_setup, "mirror_notebooks") as mirror, mock.patch.object(
            course_setup, "snapshot"
        ) as snap:
            course_setup.SyncDaemon("/tmp/repo").run_once()
            mirror.assert_called_once()
            snap.assert_called_once()

    def test_start_auto_sync_starts_daemon_when_repo_present(self):
        class FakeDaemon:
            def __init__(self, repo_root, interval=5.0):
                self.repo_root = repo_root
                self.started = False

            def start(self):
                self.started = True
                return self

        with mock.patch.object(course_setup, "_auto_sync_handle", None), mock.patch.object(
            course_setup, "find_repo_root", return_value=Path("/repo")
        ), mock.patch.object(course_setup, "SyncDaemon", FakeDaemon):
            handle = course_setup._start_auto_sync(self._env())
            self.assertIsInstance(handle, FakeDaemon)
            self.assertTrue(handle.started)

    def test_start_auto_sync_uses_exporter_on_colab(self):
        with mock.patch.object(course_setup, "_auto_sync_handle", None), mock.patch.object(
            course_setup, "find_repo_root", return_value=None
        ):
            handle = course_setup._start_auto_sync(self._env(in_colab=True, iscolab=True))
            self.assertIsInstance(handle, course_setup.ColabExporter)

    def test_start_auto_sync_noop_when_no_repo_off_colab(self):
        with mock.patch.object(course_setup, "_auto_sync_handle", None), mock.patch.object(
            course_setup, "find_repo_root", return_value=None
        ):
            self.assertIsNone(course_setup._start_auto_sync(self._env()))

    def test_start_auto_sync_is_idempotent(self):
        sentinel = object()
        with mock.patch.object(course_setup, "_auto_sync_handle", sentinel), mock.patch.object(
            course_setup, "find_repo_root"
        ) as find:
            self.assertIs(course_setup._start_auto_sync(self._env()), sentinel)
            find.assert_not_called()

    def test_lock_refused_when_held_by_another_live_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            course_setup._sync_lock_path(root).write_text(str(os.getppid()))  # a live process
            self.assertIsNone(course_setup._acquire_sync_lock(root))

    def test_lock_refused_when_already_held_by_this_process(self):
        # A second acquire in the SAME process (e.g. a stray run_sync next to the init daemon)
        # must back off, not reclaim its own live lock and let a later release delete it twice.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            first = course_setup._acquire_sync_lock(root)
            self.assertIsNotNone(first)
            self.assertIsNone(course_setup._acquire_sync_lock(root))

    def test_lock_reclaimed_when_holder_is_dead(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            lock = course_setup._sync_lock_path(root)
            lock.write_text("999999")  # a pid that is essentially never alive
            self.assertIsNotNone(course_setup._acquire_sync_lock(root))
            self.assertEqual(lock.read_text().strip(), str(os.getpid()))

    def test_acquire_then_release_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            acquired = course_setup._acquire_sync_lock(root)
            self.assertTrue(course_setup._sync_lock_path(root).exists())
            course_setup._release_sync_lock(acquired)
            self.assertFalse(course_setup._sync_lock_path(root).exists())

    def test_run_sync_once_runs_one_pass(self):
        with mock.patch.object(
            course_setup, "find_repo_root", return_value=Path("/repo")
        ), mock.patch.object(course_setup, "pull_notebooks") as pull, mock.patch.object(
            course_setup, "mirror_notebooks"
        ) as mirror, mock.patch.object(course_setup, "snapshot") as snap:
            course_setup.run_sync(once=True)
            pull.assert_called_once()
            mirror.assert_called_once()
            snap.assert_called_once()

    def test_run_sync_noop_without_repo(self):
        with mock.patch.object(course_setup, "find_repo_root", return_value=None):
            course_setup.run_sync(once=True)  # must not raise


class AutoSaverDedupTests(unittest.TestCase):
    def _training_result(self):
        result = mock.Mock()
        result.success = True
        result.info.raw_cell = "learn.fine_tune(3)"
        return result

    def setUp(self):
        course_setup._models_saved_this_cell.clear()
        self.addCleanup(course_setup._models_saved_this_cell.clear)

    def test_hook_skips_model_already_saved_this_cell(self):
        # Simulate cached_model having saved this exact object earlier in the same cell.
        with temp_drive():
            learner = FakeLearner()
            saver = course_setup.AutoSaver("lesson-1")
            course_setup._models_saved_this_cell.add(id(learner))
            with mock.patch.object(course_setup, "_user_namespace", return_value={"learn": learner}), \
                mock.patch.object(course_setup, "save_model") as fake_save:
                saver._on_post_run_cell(self._training_result())
            fake_save.assert_not_called()
        # The per-cell set is cleared afterwards so the next cell starts fresh.
        self.assertEqual(course_setup._models_saved_this_cell, set())

    def test_hook_saves_a_new_model_from_a_training_cell(self):
        with temp_drive():
            learner = FakeLearner()
            saver = course_setup.AutoSaver("lesson-1")
            with mock.patch.object(course_setup, "_user_namespace", return_value={"learn": learner}), \
                mock.patch.object(course_setup, "save_model") as fake_save:
                saver._on_post_run_cell(self._training_result())
            fake_save.assert_called_once()

    def test_hook_ignores_non_training_cell(self):
        with temp_drive():
            saver = course_setup.AutoSaver("lesson-1")
            result = mock.Mock()
            result.success = True
            result.info.raw_cell = "dls.show_batch()"
            with mock.patch.object(course_setup, "_user_namespace", return_value={"learn": FakeLearner()}), \
                mock.patch.object(course_setup, "save_model") as fake_save:
                saver._on_post_run_cell(result)
            fake_save.assert_not_called()


class RestoreTests(unittest.TestCase):
    def test_dataset_restores_to_its_recorded_location(self):
        with temp_drive(), tempfile.TemporaryDirectory() as workspace:
            folder = Path(workspace) / "pill_data" / "pill_or_not"
            (folder / "pill").mkdir(parents=True)
            (folder / "pill" / "a.txt").write_text("img")

            course_setup.save_folder(folder, "lesson-1", "pill_or_not")
            # The manifest should remember exactly where the folder lived.
            manifest = course_setup.load_manifest("lesson-1")
            self.assertEqual(manifest["artifacts"]["pill_or_not"]["restore_to"], str(folder))

            # Wipe the working copy, then restore from Drive; it must land back in `folder`.
            shutil.rmtree(folder)
            self.assertFalse(folder.exists())
            count = course_setup.restore("lesson-1", dest_root=Path(workspace) / "unused")
            self.assertGreaterEqual(count, 1)
            self.assertEqual((folder / "pill" / "a.txt").read_text(), "img")

    def test_restore_skips_missing_files_without_counting_them(self):
        # dest_root is an explicit temp dir so restore's manifest copy never lands in the
        # repo (restore defaults dest_root to cwd).
        with temp_drive(), tempfile.TemporaryDirectory() as dest:
            # Record an artifact whose file does not exist on disk.
            course_setup.record_artifact(
                "lesson-1", "ghost",
                {"kind": "model", "drive_path": "/nope/missing.pkl"},
            )
            self.assertEqual(course_setup.restore("lesson-1", dest_root=dest), 0)

    def test_relative_restore_to_resolves_against_dest_root(self):
        # The notebook saves at a cwd-relative path (e.g. 'pill_data/pill_or_not'); a
        # terminal restore into a different dest_root must land the data under that root.
        original_cwd = os.getcwd()
        with temp_drive(), tempfile.TemporaryDirectory() as workspace:
            try:
                os.chdir(workspace)
                folder = Path("pill_data") / "pill_or_not"
                (folder / "pill").mkdir(parents=True)
                (folder / "pill" / "a.txt").write_text("img")
                course_setup.save_folder(folder, "lesson-1", "pill_or_not")
                recorded = course_setup.load_manifest("lesson-1")["artifacts"]["pill_or_not"]
                self.assertFalse(Path(recorded["restore_to"]).is_absolute())
            finally:
                os.chdir(original_cwd)

            dest = Path(workspace) / "elsewhere"
            count = course_setup.restore("lesson-1", dest_root=dest)
            self.assertGreaterEqual(count, 1)
            self.assertEqual(
                (dest / "pill_data" / "pill_or_not" / "pill" / "a.txt").read_text(), "img"
            )


class SaveModelTrackingTests(unittest.TestCase):
    """The per-cell de-dup set must only be populated when the hook will clear it."""

    def setUp(self):
        self._prev_active = course_setup._autosave_hook_active
        course_setup._models_saved_this_cell.clear()
        self.addCleanup(course_setup._models_saved_this_cell.clear)
        self.addCleanup(setattr, course_setup, "_autosave_hook_active", self._prev_active)

    def test_no_tracking_when_autosave_off(self):
        course_setup._autosave_hook_active = False
        with temp_drive():
            course_setup.save_model(FakeLearner(), "lesson-1", "m")
        self.assertEqual(course_setup._models_saved_this_cell, set())

    def test_tracks_when_autosave_on(self):
        course_setup._autosave_hook_active = True
        with temp_drive():
            learner = FakeLearner()
            course_setup.save_model(learner, "lesson-1", "m")
        self.assertIn(id(learner), course_setup._models_saved_this_cell)


if __name__ == "__main__":
    unittest.main()
