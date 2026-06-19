"""Unit tests for course_setup.

These use only the standard library (``unittest`` plus ``unittest.mock``), so they run
with ``python -m unittest`` or ``python -m pytest`` without installing torch, fastai, or
any other heavy dependency. Everything those libraries would do is replaced with a mock,
which lets us assert on the decisions the module makes (which device, install or skip,
which path) rather than on real hardware or network.
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import course_setup


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


if __name__ == "__main__":
    unittest.main()
