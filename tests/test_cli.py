import json

import pytest

from arclm.cli import main
from arclm import __version__


pytestmark = pytest.mark.cli


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_cli_info_json(capsys):
    assert main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["arclm"] == __version__
    assert payload["torch"]


def test_cli_data_validate_json(tmp_path, capsys):
    path = tmp_path / "records.jsonl"
    path.write_text('{"text": "valid row"}\n{"text": ""}\n', encoding="utf-8")

    code = main(["data", "validate", str(path), "--schema", "text", "--strict", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["total_records"] == 2
    assert payload["invalid_records"] == 1


def test_cli_model_inspect_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["model", "inspect", "--help"])

    assert exc.value.code == 0
    assert "Inspect model support" in capsys.readouterr().out
