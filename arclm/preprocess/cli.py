from __future__ import annotations

import argparse
import json
from .config import PreprocessConfig
from .pipeline import PreprocessPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ArcLM dataset preprocessing")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--config", "-c", help="YAML config path")
    parser.add_argument("--report-dir", default="reports/preprocess", help="Report output directory")
    args = parser.parse_args()

    cfg = PreprocessConfig.from_yaml(args.config) if args.config else PreprocessConfig()
    report = PreprocessPipeline(cfg).run(args.input, args.output, args.report_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
