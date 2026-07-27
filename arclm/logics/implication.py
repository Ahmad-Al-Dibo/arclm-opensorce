from .sentence import Sentence


class Implication(Sentence):

    def __init__(
        self,
        antecedent,
        consequent
    ):
        Sentence.validate(antecedent)
        Sentence.validate(consequent)

        self.antecedent = antecedent
        self.consequent = consequent

    def evaluate(self, model):
        return (
            not self.antecedent.evaluate(model)
            or self.consequent.evaluate(model)
        )

    def formula(self):
        antecedent = Sentence.parenthesize(
            self.antecedent.formula()
        )

        consequent = Sentence.parenthesize(
            self.consequent.formula()
        )

        return f"{antecedent} => {consequent}"

    def symbols(self):
        return set.union(
            self.antecedent.symbols(),
            self.consequent.symbols()
        )