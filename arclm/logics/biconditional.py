from .sentence import Sentence


class Biconditional(Sentence):

    def __init__(self, left, right):
        Sentence.validate(left)
        Sentence.validate(right)

        self.left = left
        self.right = right

    def evaluate(self, model):
        return (
            self.left.evaluate(model)
            == self.right.evaluate(model)
        )

    def formula(self):
        left = Sentence.parenthesize(
            self.left.formula()
        )

        right = Sentence.parenthesize(
            self.right.formula()
        )

        return f"{left} <=> {right}"

    def symbols(self):
        return set.union(
            self.left.symbols(),
            self.right.symbols()
        )