"""Compatibility entrypoint for ArcLM's packaged simple interface.

Prefer:
    python -m arclm --run simple-interface

Or from Python:
    from arclm import run_simple_interface
    run_simple_interface()
"""

from arclm.simple_interface import app, run_simple_interface


if __name__ == "__main__":
    run_simple_interface()
