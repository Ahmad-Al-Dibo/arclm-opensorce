import json

import pytest

from arclm.cli import main


pytestmark = pytest.mark.cli


def test_phase3_data_cli_commands(tmp_path, capsys):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "1", "text": "hello"}),
                json.dumps({"id": "2", "text": "hello"}),
                json.dumps({"id": "3", "text": "world"}),
            ]
        ),
        encoding="utf-8",
    )

    assert main(["data", "analyze", str(path), "--format", "jsonl", "--json"]) == 0
    assert "duplicate_records" in capsys.readouterr().out

    assert main(["data", "split", str(path), "--format", "jsonl", "--json"]) == 0
    assert "split_report" in capsys.readouterr().out

    assert main(["data", "shard", str(path), "--format", "jsonl", "--num-shards", "2", "--json"]) == 0
    assert "shard_report" in capsys.readouterr().out

    assert main(["data", "fingerprint", str(path), "--json"]) == 0
    assert "sha256" in capsys.readouterr().out


def test_phase3_cache_runs_plugins_cli(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    assert main(["cache", "inspect", str(cache_dir)]) == 0
    assert "cache_stats" in capsys.readouterr().out

    assert main(["cache", "clear", str(cache_dir)]) == 0
    assert "cache_stats" in capsys.readouterr().out

    assert main(["runs", "list", "--output-dir", str(tmp_path / "runs"), "--json"]) == 0
    assert "[]" in capsys.readouterr().out

    assert main(["plugins", "list"]) == 0
    assert "[]" in capsys.readouterr().out
