"""Logic basics: build simple rules and ask whether a conclusion follows."""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm.logics import And, Implication, Not, Or, Symbol, model_check


def main():
    studied = Symbol("Studied")
    practiced = Symbol("Practiced")
    ready = Symbol("Ready")

    knowledge = And(
        Implication(And(studied, practiced), ready),
        studied,
        practiced,
    )

    print("Knowledge:", knowledge.formula())
    print("Can we prove Ready?", model_check(knowledge, ready))
    print("Can we prove not Ready?", model_check(knowledge, Not(ready)))

    choice = Or(studied, practiced)
    print("At least one learning action:", choice.formula())


if __name__ == "__main__":
    main()
