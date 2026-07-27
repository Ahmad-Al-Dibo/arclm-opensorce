from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arclm"))

from logics import (
    And,
    Biconditional,
    Implication,
    Not,
    Or,
    Symbol,
    model_check,
)


def main():
    p = Symbol("P")
    q = Symbol("Q")
    r = Symbol("R")

    complex_sentence = And(
        Implication(p, q),
        Or(Not(q), r),
    )

    model = {"P": True, "Q": False, "R": True}

    print("Formula:", complex_sentence.formula())
    print("Evaluation:", complex_sentence.evaluate(model))
    print("Symbols:", sorted(complex_sentence.symbols()))

    knowledge = And(Implication(p, q), p)
    query = q
    print("Model check (knowledge => query):", model_check(knowledge, query))

    biconditional = Biconditional(p, q)
    print("Biconditional formula:", biconditional.formula())
    print("Biconditional evaluation:", biconditional.evaluate({"P": True, "Q": True}))


if __name__ == "__main__":
    main()
