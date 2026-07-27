"""Setuptools entrypoint and ArcLM release helper.

Common commands:

    python setup.py --version
    python setup.py release --version 0.5.0
    python setup.py release --version 0.5.0 --upload
    python setup.py release --version 0.5.0 --readme README.md --upload
    python setup.py release --version 0.5.0 --repository testpypi --upload

    # for tagged releases:

        python setup.py release --version 0.5.0 --tag --push --upload

PyPI does not allow replacing files for an already published version.
If a file for a version already exists on PyPI, choose an unpublished version.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from setuptools import Command, setup


SUPPORTED_PYTHON = (3, 9), (3, 13)


def validate_python_version() -> None:
    if not (sys.version_info >= SUPPORTED_PYTHON[0] and sys.version_info < SUPPORTED_PYTHON[1]):
        min_ver = ".".join(str(v) for v in SUPPORTED_PYTHON[0])
        max_ver = ".".join(str(v) for v in (SUPPORTED_PYTHON[1][0], SUPPORTED_PYTHON[1][1] - 1))
        raise RuntimeError(
            f"ArcLM requires Python >= {min_ver} and < {SUPPORTED_PYTHON[1][0]}.{SUPPORTED_PYTHON[1][1]}."
            f" Current interpreter is {sys.version_info.major}.{sys.version_info.minor}."
        )


def validate_torch_installation() -> None:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "ArcLM requires torch to run tests and build. "
            "Install a supported torch wheel in your environment before running setup.py release."
        )
    if importlib.util.find_spec("torch.utils") is None:
        raise RuntimeError(
            "The installed torch package is incomplete or broken. "
            "Reinstall torch>=2.1,<3 with a valid wheel provider."
        )


validate_python_version()

ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT / "arclm" / "_version.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"
DEFAULT_README = "README.md"
VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([^"]+)"')
PLAIN_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_current_version() -> str:
    match = VERSION_PATTERN.search(VERSION_FILE.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Could not find __version__ in {VERSION_FILE}")
    return match.group(1)


def validate_version(version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+([a-zA-Z0-9._+-]+)?", version):
        raise ValueError(
            "Version should look like 0.1.0, 0.1.1, 0.2.0, or 1.0.0"
        )


def bump_version(version: str, part: str) -> str:
    """Bump a plain semantic version."""
    match = PLAIN_VERSION_PATTERN.fullmatch(version)
    if not match:
        raise ValueError("--bump requires the current version to look like X.Y.Z")

    major, minor, patch = (int(value) for value in match.groups())
    normalized = part.lower().strip()
    if normalized == "patch":
        patch += 1
    elif normalized == "minor":
        minor += 1
        patch = 0
    elif normalized == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError("--bump must be one of: patch, minor, major")
    return f"{major}.{minor}.{patch}"


def update_version_file(version: str) -> None:
    text = VERSION_FILE.read_text(encoding="utf-8")
    new_text, count = VERSION_PATTERN.subn(f'__version__ = "{version}"', text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update __version__ in {VERSION_FILE}")
    VERSION_FILE.write_text(new_text, encoding="utf-8")


def update_pyproject_readme(readme: str) -> None:
    readme_path = (ROOT / readme).resolve()
    if not readme_path.exists():
        raise FileNotFoundError(f"Readme file does not exist: {readme}")
    if ROOT not in readme_path.parents and readme_path != ROOT:
        raise ValueError("Readme must be inside the project directory")

    relative_readme = readme_path.relative_to(ROOT).as_posix()
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(?m)^readme\s*=\s*"[^"]+"',
        f'readme = "{relative_readme}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not update readme in {PYPROJECT_FILE}")
    PYPROJECT_FILE.write_text(new_text, encoding="utf-8")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def pypi_json_url(repository: str) -> Optional[str]:
    """Return the JSON API URL for known PyPI repositories."""
    normalized = (repository or "pypi").lower().strip().rstrip("/")
    if normalized in {"pypi", "https://upload.pypi.org/legacy"}:
        return "https://pypi.org/pypi/arclm/json"
    if normalized in {"testpypi", "test-pypi", "https://test.pypi.org/legacy"}:
        return "https://test.pypi.org/pypi/arclm/json"
    return None


def fetch_published_versions(repository: str) -> set[str]:
    """Fetch versions already published for the configured repository."""
    url = pypi_json_url(repository)
    if url is None:
        print(
            f"Skipping published-version check for custom repository: {repository}",
            flush=True,
        )
        return set()

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise RuntimeError(f"Could not check published versions at {url}: {exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc

    return set(payload.get("releases", {}))


def plain_version_tuple(version: str) -> Optional[tuple[int, int, int]]:
    """Return a comparable tuple for plain X.Y.Z versions."""
    match = PLAIN_VERSION_PATTERN.fullmatch(version)
    if not match:
        return None
    return tuple(int(value) for value in match.groups())


def latest_plain_version(
    versions: set[str],
) -> Optional[tuple[str, tuple[int, int, int]]]:
    """Return the latest plain semantic version from a version set."""
    parsed = [
        (version, plain_version_tuple(version))
        for version in versions
    ]
    parsed = [(version, value) for version, value in parsed if value is not None]
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[1])


def ensure_version_can_be_uploaded(
    version: str,
    repository: str,
    skip_existing: bool,
) -> None:
    """Fail early if the target package version already exists."""
    published_versions = fetch_published_versions(repository)
    if version in published_versions:
        message = (
            f"ArcLM {version} is already published on {repository}. "
            "PyPI does not allow overwriting or updating files for the same version. "
            "Use --bump patch, --bump minor, or pass a new --version value."
        )
        if skip_existing:
            print(f"[WARNING] {message} Twine will skip existing files.", flush=True)
            return
        raise RuntimeError(message)

    latest = latest_plain_version(published_versions)
    target = plain_version_tuple(version)
    if latest is not None and target is not None:
        latest_version, latest_tuple = latest
        if target <= latest_tuple:
            suggested = f"{latest_tuple[0]}.{latest_tuple[1]}.{latest_tuple[2] + 1}"
            raise RuntimeError(
                f"ArcLM {latest_version} is already the latest published version on "
                f"{repository}. Refusing to upload older version {version}. "
                f"Choose a newer version such as --version {suggested}."
            )


class ReleaseCommand(Command):
    """Update version/readme, build distributions, and optionally upload."""

    description = "build and optionally upload a new ArcLM release"
    user_options = [
        ("version=", None, "package version, for example 0.5.0"),
        ("bump=", None, "bump current version: patch, minor, or major"),
        ("readme=", None, "readme file to use in pyproject.toml"),
        ("repository=", None, "twine repository name or URL, default: pypi"),
        ("upload", None, "upload dist files with twine"),
        ("skip-existing", None, "pass --skip-existing to twine upload"),
        ("no-pypi-version-check", None, "skip the pre-upload PyPI duplicate-version check"),
        ("tag", None, "create an annotated git tag named vVERSION"),
        ("push", None, "push the current branch and release tag"),
        ("skip-tests", None, "skip pytest before building"),
        ("keep-dist", None, "keep existing files in dist/ instead of cleaning it"),
    ]
    boolean_options = [
        "upload",
        "skip-existing",
        "no-pypi-version-check",
        "tag",
        "push",
        "skip-tests",
        "keep-dist",
    ]

    def initialize_options(self) -> None:
        self.version = None
        self.bump = None
        self.readme = DEFAULT_README
        self.repository = "pypi"
        self.upload = False
        self.skip_existing = False
        self.no_pypi_version_check = False
        self.tag = False
        self.push = False
        self.skip_tests = False
        self.keep_dist = False

    def finalize_options(self) -> None:
        if self.bump is not None and self.version is not None:
            raise ValueError("Use either --version or --bump, not both.")
        if self.bump is not None:
            self.version = bump_version(read_current_version(), self.bump)
        elif self.version is None:
            self.version = read_current_version()
        validate_version(self.version)
        if self.push:
            self.tag = True

    def run(self) -> None:
        if self.upload and not self.no_pypi_version_check:
            ensure_version_can_be_uploaded(
                self.version,
                self.repository,
                self.skip_existing,
            )

        update_version_file(self.version)
        update_pyproject_readme(self.readme)

        if not self.skip_tests:
            run([sys.executable, "-m", "pytest", "tests"])

        if not self.keep_dist:
            shutil.rmtree(ROOT / "dist", ignore_errors=True)
            shutil.rmtree(ROOT / "build", ignore_errors=True)

        run([sys.executable, "-m", "build"])
        run([sys.executable, "-m", "twine", "check", "dist/*"])

        if self.tag:
            tag_name = f"v{self.version}"
            run(["git", "add", "arclm/_version.py", "pyproject.toml"])
            run(["git", "commit", "-m", f"Release ArcLM {self.version}"])
            run(["git", "tag", "-a", tag_name, "-m", f"ArcLM {self.version}"])

        if self.upload:
            upload_command = [
                sys.executable,
                "-m",
                "twine",
                "upload",
                "--repository",
                self.repository,
            ]
            if self.skip_existing:
                upload_command.append("--skip-existing")
            upload_command.append("dist/*")
            run(upload_command)
        else:
            print("Built and checked dist files. Add --upload to publish them.")

        if self.push:
            run(["git", "push"])
            run(["git", "push", "origin", f"v{self.version}"])


setup(
    name="arclm",
    version=read_current_version(),
    cmdclass={"release": ReleaseCommand},
)
