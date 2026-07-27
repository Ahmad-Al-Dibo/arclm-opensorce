from .sentence import Sentence
from .exceptions import EvaluationException


class Symbol(Sentence):

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def __hash__(self):
        return hash(("symbol", self.name))

    def __repr__(self):
        return self.name

    def evaluate(self, model):
        try:
            return bool(model[self.name])
        except KeyError:
            raise EvaluationException(
                f"variable {self.name} not in model"
            )

    def formula(self):
        return self.name

    def symbols(self):
        return {self.name}