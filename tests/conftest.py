import contextlib
import os
import shutil
import tempfile
from pathlib import Path
import typing as t
from traceback import print_tb

import pytest
from typer.testing import CliRunner as BaseCliRunner

from domteur.config import Settings, CONFIG_FN, ENV_CONFIG_KEY


class CliRunner(BaseCliRunner):

    with_traceback = True

    def invoke(self, cli, commands, **kwargs):
        result = super(CliRunner, self).invoke(cli, commands, **kwargs)
        if not result.exit_code == 0 and self.with_traceback:
            print_tb(result.exc_info[2])
            print(result.exception)
        return result



@contextlib.contextmanager
def cd_to_directory(path: Path, env: t.Optional[dict] = None):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    if env:
        for k, v in env.items():
            os.environ[k] = v
    try:
        yield
    finally:
        os.chdir(prev_cwd)
        if env:
            for k in env:
                os.environ[k] = ""


@pytest.fixture(scope="function")
def temporary_directory():
    """Provides a temporary directory that is removed after the test."""
    directory = tempfile.mkdtemp()
    yield Path(directory)
    shutil.rmtree(directory)


@pytest.fixture()
def cfg_file(temporary_directory):
    s = Settings(file_path=temporary_directory / CONFIG_FN)
    s.save()
    return s


@pytest.fixture()
def runner(temporary_directory, cfg_file):
    with cd_to_directory(
        temporary_directory, env={ENV_CONFIG_KEY: str(temporary_directory)}
    ):
        yield CliRunner()
