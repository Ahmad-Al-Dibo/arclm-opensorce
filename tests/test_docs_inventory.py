from pathlib import Path

import pytest


pytestmark = pytest.mark.docs


REQUIRED_DOCS = [
    "docs/index.md",
    "docs/quick-start.md",
    "docs/supported-models.md",
    "docs/api-reference/index.md",
    "docs/cli-reference.md",
    "docs/migration-guide.md",
    "docs/production-readiness.md",
    "docs/audit/repository-audit.md",
    "docs/audit/public-api-inventory.md",
    "docs/audit/documentation-gaps.md",
]


def test_required_documentation_files_exist():
    for path in REQUIRED_DOCS:
        assert Path(path).exists(), path


def test_supported_models_doc_uses_support_levels():
    text = Path("docs/supported-models.md").read_text(encoding="utf-8")

    assert "Officially supported" in text
    assert "Experimentally supported" in text
    assert "Compatible but not tested" in text
    assert "Not supported" in text
