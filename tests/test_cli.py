from pathlib import Path

from domteur.config import CONFIG_FN
from domteur.main import cli


def test_help(runner, temporary_directory):
    assert "/tmp" in str(Path().resolve()), (
        "The runner fixture must be in context of a tempdir"
    )
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert (temporary_directory / CONFIG_FN).is_file()
    result = runner.invoke(cli, ["show-config"])
    assert result.exit_code == 0
