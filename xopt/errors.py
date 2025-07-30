class XoptError(Exception):
    pass


class EvaluatorError(Exception):
    pass


class GeneratorError(Exception):
    pass


class SeqGeneratorError(GeneratorError):
    pass


class VOCSError(Exception):
    pass


class XoptWarning(Warning):
    pass


class GeneratorWarning(XoptWarning):
    pass


class FeasibilityError(Exception):
    pass
