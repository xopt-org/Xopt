class XoptError(Exception):
    pass


class EvaluatorError(Exception):
    pass


class GeneratorError(Exception):
    pass


class SeqGeneratorError(GeneratorError):
    pass


class VOCSError(Exception):
    """
    Exception for when VOCS are invalid for the generator being initialized.
    """


class DataError(Exception):
    """
    Exception related to data being passed to Xopt or Generator objects from user.
    """


class XoptWarning(Warning):
    pass


class GeneratorWarning(XoptWarning):
    pass


class FeasibilityError(Exception):
    pass
