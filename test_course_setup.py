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
            mock.patch.object(course_setup, "select_device", manager.select_device):
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
            mock.patch.object(course_setup, "select_device", return_value="device:cpu"):
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


class DetectModelsTests(unittest.TestCase):
    def test_training_cell_with_learner_is_detected(self):
        learner = FakeLearner()
        found = course_setup.detect_models_to_save(
            "learn.fine_tune(3)", {"learn": learner, "x": 5}
        )
        self.assertEqual(found, [("learn", learner)])

    def test_non_training_cell_detects_nothing(self):
        found = course_setup.detect_models_to_save(
            "dls.show_batch()", {"learn": FakeLearner()}
        )
        self.assertEqual(found, [])

    def test_underscored_names_are_ignored(self):
        found = course_setup.detect_models_to_save(
            "learn.fit_one_cycle(1)", {"_hidden": FakeLearner()}
        )
        self.assertEqual(found, [])


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
