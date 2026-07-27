"""Allow running ArcLM with ``python -m arclm``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
