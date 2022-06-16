from collections.abc import Sequence

from deap import base


# New version, uses inheritance
class FitnessWithConstraints2(base.Fitness):
    """
    Modification of base Fitness class to include constraints

    Feasibility is defined as: all constraints <= 0

    """

    def __init__(self, values=(), constraints=None):
        if self.weights is None:
            raise TypeError(
                "Can't instantiate abstract %r with abstract "
                "attribute weights." % (self.__class__)
            )

        if not isinstance(self.weights, Sequence):
            raise TypeError(
                "Attribute weights of %r must be a sequence." % self.__class__
            )

        if len(values) > 0:
            self.values = values
        if constraints:
            self.cvalues = constraints
        else:
            self.cvalues = None

    def feasible(self):
        if not self.cvalues:
            return True
        """A feasible solution must have all constraints <= 0"""
        if any([x > 0 for x in self.cvalues]):
            return False
        else:
            return True

    def dominates(self, other, obj=slice(None)):
        """A feasible solution must have all constraints <= 0"""
        if not self.cvalues:
            return self.old_dominates(other, obj=obj)
        f1 = self.feasible()
        f2 = other.feasible()
        if f1 and f2:
            return self.old_dominates(other, obj=obj)
        elif f1 and not f2:
            return True
        elif f2 and not f1:
            return False
        else:
            # Both infeasible
            is_better = False
            for self_cvalue, other_cvalue in zip(self.cvalues, other.cvalues):
                if self_cvalue <= 0 and other_cvalue <= 0:
                    continue
                if self_cvalue > other_cvalue:
                    return False
                elif self_cvalue < other_cvalue:
                    return True
            return is_better

    def old_dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        copy_.cvalues = self.cvalues
        return copy_
