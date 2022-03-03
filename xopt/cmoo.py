from packaging import version
import pymoo

from . import vocs_tools
from .tools import DummyExecutor


DEPRECATED_PYMOO = True
if version.parse(pymoo.__version__) >= version.parse("0.5.0"):
    # pymoo reorganized the project with 0.5.0
    # see more here: https://github.com/anyoptimization/pymoo/releases/tag/0.5.0
    # from pymoo.core.problem import ElementwiseProblem as Problem
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
    DEPRECATED_PYMOO = False
else:
    # Backwards compatibility with < 0.5.0
    from pymoo.model.problem import Problem
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.operators.sampling.random_sampling import FloatRandomSampling
    from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover

from pymoo.optimize import minimize

import autograd.numpy as anp
import numpy as np

#------------------------------------------
# Additions to pymoo

def dummy_f(n_var=2, n_obj=2, n_constr=2):
    
    res = {}
    a = 1e15
    res['X'] = n_var*[a]
    res['F'] = n_obj*[a]
    res['G'] = n_constr*[a]
    
    return res


class ContinuousProblem(Problem):
    """
    Modification to pymoo's Problem class to suppor rolling, continuous evaluation.

    """
    def __init__(self,
        n_var=-1,
        n_obj=-1,
        n_constr=0,
        xl=None,
        xu=None,
        type_var=np.double,
        evaluation_of="auto",
        replace_nan_values_of="auto",
        parallelization=None,
        elementwise_evaluation=True,
        callback=None,
        # new arguments
        evaluate_f=None,         
        archive_callback=None,
        rolling_evaluate=True):  

        if DEPRECATED_PYMOO:
            # Require this
            assert elementwise_evaluation, 'Must have elementwise_evaluation for continuous evaluation '

        init_args = dict(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            type_var=type_var,
            evaluation_of=evaluation_of,
            # replace_nan_values_of= replace_nan_values_of,
            parallelization=parallelization,
            elementwise_evaluation=elementwise_evaluation,
            callback=callback
        )

        if not DEPRECATED_PYMOO:
            init_args.pop('elementwise_evaluation')

        # Try to make as similar to normal routine
        super().__init__(**init_args)
        
        if not parallelization:
            self.executor = DummyExecutor()
        else:
            assert parallelization[0] == 'futures'
            self.executor = parallelization[1] 

        self.rolling_evaluate = rolling_evaluate
        self.f = evaluate_f
        self.e_callback = archive_callback
        
        
        # Internal
        self.futures = []
    
    def submit(self, X):
        """
        Submit jobs
        """
        
        return [self.executor.submit(self.f, x) for x in X]
    
    def submit_dummies(self, n):
        """
        Submit n dummy jobs
        """
        return [self.executor.submit(dummy_f, n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr) for i in range(n)]    

    def _evaluate(self, X, out, *args, **kwargs):
        return self._evaluate_elementwise(X=X, out=out, calc_gradient=None)

    #def _evaluate(self, X, out, *args, **kwargs):
    def _evaluate_elementwise(self, X, calc_gradient, out, *args, **kwargs):
        """
        Rolling evaluate.
        
        Note that evaluate_f should assign X as well as F, G
        
        """
        results = []
        
        # for testing
        # np.random.shuffle(X)
        
        # If no futures are already there, just evaluate all at once
        if not self.futures:
            print('evaluating all at once', X.shape)
            self.futures = self.submit(X)
            results = [fut.result() for fut in self.futures]
            done = True
            
            # Submit dummies
            if self.rolling_evaluate:
                self.futures = self.submit_dummies(len(X))
            else:
                self.futures = []
            
        else:
            done = False
        
        # Continuous loop 
        imax = len(X)-1
        i0 = 0 # index of item in X. This will be iterated over until X is fully submitted.  
        while not done:    
            for ix, fut in enumerate(self.futures):
                if not fut.done():
                    continue
                res = fut.result()   
                assert 'X' in res, 'evaluate_f must have X in its output'
                results.append(res)
                self.futures[ix] = self.executor.submit(self.f, X[i0]) # send off new work                                
                
                # exit when all of X has been submitted
                if i0 == imax:
                    done = True 
                    break
                else:
                    i0 += 1
        
        # stack all the single outputs together        
        for key in results[0].keys():
            out[key] = anp.row_stack([results[i][key] for i in range(len(results))])
            
        return out 
    
    
    
#------------------------------------------
# Xopt custom with VOCS things
    

def cmoo_evaluate(vec, labeled_evaluate_f=None, vocs=None, include_inputs_and_outputs=True, verbose=True):
    """
    Evaluation function wrapper for use with cngsa. Returns dict with:
        'vec', 'obj', 'con', 'err'
    
    If a vocs is given, the function evaluate_f is assumed to have labeled inputs and outputs,
    and vocs will be used to form the output above. If include_inputs_and_outputs, then:
        'inputs', 'outputs'
    will be included in the returned dict. 
    
    Otherwise, evaluate_f should return pure numbers as:
        vec -> (objectives, constraints)

    This function will be evaluated by a worker.
    
    Any exceptions will be caught, and this will return:
        error = True
        0 for all objectives
        -666.0 for all constraints

    """
    
    result = {}
    
    
    if vocs:
        # labeled inputs -> labeled outputs evaluate_f
        inputs = vocs_tools.inputs_from_vec(vec, vocs=vocs) 
    
    try:
    
        if vocs:
            
            # Evaluation
            inputs0 = inputs.copy()       # Make a copy, because the evaluate function might modify the inputs.
            outputs = labeled_evaluate_f(inputs0)
        
            obj_eval = vocs_tools.evaluate_objectives(vocs.objectives, outputs)
            con_eval = vocs_tools.evaluate_constraints(vocs.constraints, outputs)
            
        else:
        # Pure number function
            obj_eval, con_eval = evaluate_f(vec)
       
        # Form numpy arrays with correct signs
        obj_eval = -np.array(vocs_tools.weight_list(vocs.objectives))*np.array(obj_eval)
        con_eval = -1*np.array(con_eval)
    
        err = False
    
    
    except Exception as ex:
        if verbose:
            print('Exception caught in cmoo_evaluate:', ex)
        outputs = {'Exception':  str(ex)}
        
        # Dummy output
        err = True
        obj_eval = np.full(len(vocs.objectives), 1e30) 
        con_eval = np.full(len(vocs.constraints), 1e30) 

    finally:
         # Add these to result
        if include_inputs_and_outputs:
            result['inputs'] = inputs
            result['outputs'] = outputs
        
    
    result['X'] = vec
    result['F'] = obj_eval
    result['G'] = con_eval
    result['Error'] = err
    
    return result



def continuous_problem_init(vocs):
    var, obj, constr = vocs.variables, vocs.objectives, vocs.constraints
    
    init = {}
    init['n_var'] = len(var)
    init['n_obj'] = len(obj)
    init['n_constr'] = len(constr)
    init['xl'] = np.array(vocs_tools.var_mins(var))
    init['xu'] = np.array(vocs_tools.var_maxs(var))
    
    return init



from deap.base import Toolbox

def cnsga2(executor=None,
          vocs=None,
          population=None,
          seed=None,
          evaluate_f=None,
          output_path=None,
          max_generations = 2,
          population_size = 4,
          crossover_probability = 0.9,
          #mutation_probability = 1.0,
          #selection='auto',
           rolling_evaluate=True,
          verbose=True):
    """
    
    CNSGA using pymoo
    
    
    """
    
    
    # Problem
    #def f(x):
    #    return cmoo_evaluate(x, vocs=vocs, labeled_evaluate_f=evaluate_f)    

    ## Above doesn't work, but Toolbox does???
    toolbox = Toolbox()    
    toolbox.register('f', cmoo_evaluate, labeled_evaluate_f=evaluate_f, vocs=vocs, verbose=verbose)    
    
    init = continuous_problem_init(vocs)
    init['evaluate_f'] = toolbox.f
    if executor:
        init['parallelization'] = ('futures', executor)    
    init['rolling_evaluate'] = rolling_evaluate
    problem = ContinuousProblem(**init)
    
    # Algorithm
    if population is None:
        sampling = FloatRandomSampling()
    else:
        sampling=population

    
    algorithm =  NSGA2(
                     pop_size=population_size,
                    sampling=sampling, 
                # selection=TournamentSelection(func_comp=binary_tournament),
                     crossover=SimulatedBinaryCrossover(eta=15, prob=crossover_probability) )
                # mutation=PolynomialMutation(prob=None, eta=20),
                # eliminate_duplicates=True,
                # n_offsprings=None,
                # display=MultiObjectiveDisplay(),
    
    
    # Minimize
    
    res = minimize(problem,
               algorithm,
               ('n_gen', max_generations),
               seed=seed,
               verbose=False)
    
    return problem, res
    
    
            
            
            