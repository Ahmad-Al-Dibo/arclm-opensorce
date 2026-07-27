from .sentence import Sentence


class Or(Sentence):

    def __init__(self, *disjuncts):
        for disjunct in disjuncts:
            Sentence.validate(disjunct)

        self.disjuncts = list(disjuncts)

    def __eq__(self, other):
        return (
            isinstance(other, Or)
            and self.disjuncts == other.disjuncts
        )

    def __hash__(self):
        return hash(
            (
                "or",
                tuple(hash(d) for d in self.disjuncts)
            )
        )

    def __repr__(self):
        disjuncts = ", ".join(
            str(d) for d in self.disjuncts
        )
        return f"Or({disjuncts})"

    def evaluate(self, model):
        return any(
            d.evaluate(model)
            for d in self.disjuncts
        )

    def formula(self):
        if len(self.disjuncts) == 1:
            return self.disjuncts[0].formula()

        return " ∨ ".join(
            Sentence.parenthesize(d.formula())
            for d in self.disjuncts
        )

    def symbols(self):
        return set.union(
            *[
                d.symbols()
                for d in self.disjuncts
            ]
        )