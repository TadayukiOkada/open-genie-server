"""Checks on pyproject.toml — the parts that are expensive to change later.

The distribution name, the console-script name and the extras names get baked
into other people's install commands, systemd units and Dockerfiles, so they
are pinned here rather than left to a reviewer's memory. The rest catches the
two ways this file rots: a version that no longer matches the package, and a
module that stops being shipped because it moved out of src/.

No install is needed — everything here reads the source tree.
"""

from pathlib import Path

import pytest

import genie_server

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 — pytest brings tomli there
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject():
    if tomllib is None:
        pytest.skip("no TOML parser: needs Python 3.11+ or tomli")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_distribution_and_import_names(pyproject):
    # The distribution is open-genie-server; the import package stays
    # genie_server, which is what every doc, deployment and `uvicorn
    # genie_server.asgi:app` invocation says.
    assert pyproject["project"]["name"] == "open-genie-server"
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert (REPO_ROOT / "src" / "genie_server" / "__init__.py").is_file()


def test_console_script_target_exists(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts == {"genie-server": "genie_server.cli:main"}

    module, _, attr = scripts["genie-server"].partition(":")
    from importlib import import_module
    assert callable(getattr(import_module(module), attr))


def test_launcher_shim_calls_the_same_entry_point():
    # genie-server.py is what the device deployment starts and what
    # `pgrep -f '[g]enie-server.py'` matches, so it has to keep existing and
    # keep delegating rather than growing its own copy of the CLI.
    shim = (REPO_ROOT / "genie-server.py").read_text(encoding="utf-8")
    assert "from genie_server.cli import main" in shim
    assert "argparse" not in shim


def test_version_is_single_sourced(pyproject):
    assert pyproject["project"]["dynamic"] == ["version"]
    version = pyproject["tool"]["setuptools"]["dynamic"]["version"]
    assert version == {"attr": "genie_server.__version__"}
    assert genie_server.__version__


def test_changelog_knows_the_current_version():
    changelog = (REPO_ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {genie_server.__version__}" in changelog


def test_core_dependencies(pyproject):
    # tokenizers is core on purpose: without it token counts fall back to
    # len(text.split()), which feeds the context check and the default
    # max_tokens, not just the reported usage. Demoting it to an extra would
    # hand that trap to anyone who runs a plain `pip install`.
    assert set(pyproject["project"]["dependencies"]) == {
        "fastapi", "uvicorn", "tokenizers"}


def test_extras(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    assert set(extras) == {"logprobs", "vlm", "test"}
    # numpy carries both logprobs and VLM; pillow is VLM-only.
    assert extras["logprobs"] == ["numpy"]
    assert set(extras["vlm"]) == {"numpy", "pillow"}
    assert {"pytest", "httpx", "requests", "jsonschema"} <= set(extras["test"])


def test_requirements_txt_defers_to_pyproject():
    # Kept only so `pip install -r requirements.txt` still works; if it grows
    # its own package list again the two will drift, which is what the move to
    # pyproject.toml was for.
    lines = [ln.strip() for ln in
             (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    assert [ln for ln in lines if ln and not ln.startswith("#")] == [".[logprobs,vlm]"]


def test_every_module_ships():
    # A module dropped outside src/ would import fine in a checkout (the tests
    # and the launcher both put src/ on sys.path) and be missing from the
    # wheel — the failure src layout exists to catch.
    package_dir = REPO_ROOT / "src" / "genie_server"
    on_disk = {p.stem for p in package_dir.glob("*.py")}
    assert "cli" in on_disk and "app" in on_disk
    assert not list(REPO_ROOT.glob("genie_server/*.py"))
