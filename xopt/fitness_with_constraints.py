

import sys

from collections import Sequence
from copy import deepcopy
from operator import mul, truediv

class FitnessWithConstraints(object):
    """The fitness is a measure of quality of a solution. If *values* are
    provided as a tuple, the fitness is initalized using those values,
    otherwise it is empty (or invalid).
    
    :param values: The initial values of the fitness as a tuple, optional.

    Fitnesses may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
    ``!=``. The comparison of those operators is made lexicographically.
    Maximization and minimization are taken care off by a multiplication
    between the :attr:`weights` and the fitness :attr:`values`. The comparison
    can be made between fitnesses of different size, if the fitnesses are
    equal until the extra elements, the longer fitness will be superior to the
    shorter.

    Different types of fitnesses are created in the :ref:`creating-types`
    tutorial.

    .. note::
       When comparing fitness values that are **minimized**, ``a > b`` will
       return :data:`True` if *a* is **smaller** than *b*.
    """
    
    weights = None
    """The weights are used in the fitness comparison. They are shared among
    all fitnesses of the same type. When subclassing :class:`Fitness`, the
    weights must be defined as a tuple where each element is associated to an
    objective. A negative weight element corresponds to the minimization of
    the associated objective and positive weight to the maximization.

    .. note::
        If weights is not defined during subclassing, the following error will 
        occur at instantiation of a subclass fitness object: 
        
        ``TypeError: Can't instantiate abstract <class Fitness[...]> with
        abstract attribute weights.``
    """
    
    wvalues = ()
    cvalues = ()
    """Contains the weighted values of the fitness, the multiplication with the
    weights is made when the values are set via the property :attr:`values`.
    Multiplication is made on setting of the values for efficiency.
    
    Generally it is unnecessary to manipulate wvalues as it is an internal
    attribute of the fitness used in the comparison operators.
    """
    
    def __init__(self, values=(), constraints=()):
        if self.weights is None:
            raise TypeError("Can't instantiate abstract %r with abstract "
                "attribute weights." % (self.__class__))
        
        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence." 
                % self.__class__)
        
        if len(values) > 0:
            self.values = values
        if len(constraints) >0:
            self.cvalues = constraints    
            self.n_constraints = len(constraints)
        
    def getValues(self):
        return tuple(map(truediv, self.wvalues, self.weights))
            
    def setValues(self, values):
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            _, _, traceback = sys.exc_info()
            raise TypeError("Both weights and assigned values must be a "
            "sequence of numbers when assigning to values of "
            "%r. Currently assigning value(s) %r of %r to a fitness with "
            "weights %s."
            % (self.__class__, values, type(values), self.weights)).with_traceback(traceback)
            
    def delValues(self):
        self.wvalues = ()

    values = property(getValues, setValues, delValues,
        ("Fitness values. Use directly ``individual.fitness.values = values`` "
         "in order to set the fitness and ``del individual.fitness.values`` "
         "in order to clear (invalidate) the fitness. The (unweighted) fitness "
         "can be directly accessed via ``individual.fitness.values``."))

    def feasible(self):
        if self.n_constraints == 0:
            return True
        """A feasible solution must have all constraints >= 0"""
        if any([x < 0 for x in self.cvalues]):
          return False
        else:
          return True

    def dominates(self, other, obj=slice(None)):
        """A feasible solution must have all constraints >= 0"""
        if self.n_constraints == 0:
            return self.old_dominates(other, obj=obj)
        f1 = self.feasible()
        f2 = other.feasible()
        if (f1 and f2):
            return self.old_dominates(other, obj=obj)
        elif (f1 and not f2):
            return True
        elif (f2 and not f1):
            return False
        else:
            # Both infeasible
            is_better = False
            for self_cvalue, other_cvalue in zip(self.cvalues, other.cvalues):
                if (self_cvalue >= 0 and other_cvalue >= 0):
                    continue
                if (self_cvalue < other_cvalue):
                    return False
                elif (self_cvalue > other_cvalue):
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

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0
        
    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)
        
    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.
        
        It assumes that the elements in the :attr:`values` tuple are 
        immutable and the fitness does not contain any other object 
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__()
        copy_.wvalues = self.wvalues
        copy_.cvalues = self.cvalues
        copy_.n_constraints = self.n_constraints
        return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
            self.values if self.valid else tuple())

