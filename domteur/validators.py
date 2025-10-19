from pathlib import Path


def string_or_path(pathlike) -> Path | None:
    if not pathlike:
        return pathlike
    if isinstance(pathlike, str):
        return Path(pathlike)
    assert isinstance(pathlike, Path), f'{pathlike} is not a path'
    return pathlike
