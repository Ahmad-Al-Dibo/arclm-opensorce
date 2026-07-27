from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def write_json_report(path: str | Path, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html_report(path: str | Path, data: Dict[str, Any]) -> None:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.get("reasons", {}).items())
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>ArcLM Preprocess Report</title></head>
<body>
<h1>ArcLM Preprocess Report</h1>
<p>Total: {data.get('total')}</p>
<p>Kept: {data.get('kept')}</p>
<p>Dropped: {data.get('dropped')}</p>
<p>Drop rate: {data.get('drop_rate'):.2%}</p>
<h2>Drop reasons</h2>
<table border='1' cellpadding='6'><tr><th>Reason</th><th>Count</th></tr>{rows}</table>
</body></html>"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
