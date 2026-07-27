def model_check(knowledge, query):

    def check_all(
        knowledge,
        query,
        symbols,
        model
    ):

        if not symbols:

            if knowledge.evaluate(model):
                return query.evaluate(model)

            return True

        remaining = symbols.copy()
        p = remaining.pop()

        model_true = model.copy()
        model_true[p] = True

        model_false = model.copy()
        model_false[p] = False

        return (
            check_all(
                knowledge,
                query,
                remaining,
                model_true
            )
            and
            check_all(
                knowledge,
                query,
                remaining,
                model_false
            )
        )

    symbols = set.union(
        knowledge.symbols(),
        query.symbols()
    )

    return check_all(
        knowledge,
        query,
        symbols,
        {}
    )