class Sentence:

    def evaluate(self, model):
        raise Exception("nothing to evaluate")

    def formula(self):
        return ""

    def symbols(self):
        return set()

    @classmethod
    def validate(cls, sentence):
        if not isinstance(sentence, Sentence):
            raise TypeError("must be a logical sentence")

    @classmethod
    def parenthesize(cls, s):

        def balanced(s):
            count = 0

            for c in s:
                if c == "(":
                    count += 1
                elif c == ")":
                    if count <= 0:
                        return False
                    count -= 1

            return count == 0

        if (
            not len(s)
            or s.isalpha()
            or (
                s[0] == "("
                and s[-1] == ")"
                and balanced(s[1:-1])
            )
        ):
            return s

        return f"({s})"