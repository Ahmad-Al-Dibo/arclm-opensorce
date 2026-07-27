from .sentence import Sentence


class And(Sentence):

    def __init__(self, *conjuncts):
        for conjunct in conjuncts:
            Sentence.validate(conjunct)

        self.conjuncts = list(conjuncts)

    def __eq__(self, other):
        return (
            isinstance(other, And)
            and self.conjuncts == other.conjuncts
        )

    def __hash__(self):
        return hash(
            (
                "and",
                tuple(hash(c) for c in self.conjuncts)
            )
        )

    def __repr__(self):
        conjunctions = ", ".join(
            str(c) for c in self.conjuncts
        )
        return f"And({conjunctions})"

    def add(self, conjunct):
        Sentence.validate(conjunct)
        self.conjuncts.append(conjunct)

    def evaluate(self, model):
        return all(
            c.evaluate(model)
            for c in self.conjuncts
        )

    def formula(self):
        if len(self.conjuncts) == 1:
            return self.conjuncts[0].formula()

        return " ∧ ".join(
            Sentence.parenthesize(c.formula())
            for c in self.conjuncts
        )

    def symbols(self):
        return set.union(
            *[
                c.symbols()
                for c in self.conjuncts
            ]
        )