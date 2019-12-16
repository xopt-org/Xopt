from .xopt import Xopt
from .tools import xopt_logo


import os
# Used to access examples directory
root, _ = os.path.split(__file__)
examples_dir = os.path.join(root, '../examples/')