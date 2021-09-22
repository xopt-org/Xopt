import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
import matlab.engine

BOUND_LOW, BOUND_UP = [0.0, 0.0], [3.14159, 3.14159]
X_RANGE = [0, 1.4]
Y_RANGE = [0, 1.4]

# labeled version
def evaluate_TNK(inputs, matlab_f='testeval', nargout=2, verbose=False):
    info = {'some': 'info', 'about': ['the', 'run']}

    x1, x2 = inputs['x1'], inputs['x2']
    
    objectives = (x1, x2)
      
    # Matlab wrapper
    rank = comm.Get_rank()
    names = matlab.engine.find_matlab()
    eng_id = names[rank]
    eng = matlab.engine.connect_matlab(eng_id)
    if verbose:
        print(f'Connecting rank {rank} to engine {eng_id}')
    
    f = getattr(eng, matlab_f)
    constraints = f(x1, x2, nargout=nargout)
    
    
    # Bounds check

    if x1 > BOUND_UP[0]:
        raise ValueError(f'Input greater than {BOUND_UP[0]} ')

    # Form outputs
    outputs = {'y1': objectives[0], 'y2': objectives[1],
               'c1': constraints[0], 'c2': constraints[1]}
    
    return outputs
