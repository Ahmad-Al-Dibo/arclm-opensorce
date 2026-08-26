from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_documented_examples_run(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    examples = sorted((root / "examples").glob("*.py"))

    assert examples
    for path in examples:
        module = _load_example(path)
        if "professional" in path.name or "lora" in path.name or "arcmodel" in path.name:
            result = module.run(tmp_path / path.stem)
        else:
            result = module.run()
        assert isinstance(result, dict)
        assert result
