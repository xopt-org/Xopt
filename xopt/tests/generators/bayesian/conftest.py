import pandas as pd

# https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html
# Note that this majorly changes the behavior of pandas - we need to prepare for 3.0
# For example, to_numpy() will return a read-only view of the data unless copy is requested
# Torch will then complain that: The given NumPy array is not writable, and PyTorch does not support non-writable
# tensors. This means writing to this tensor will result in undefined behavior.
# Solution is to either request a copy or mark array writable if dataframe is to be discarded
pd.options.mode.chained_assignment = "raise"
pd.options.mode.copy_on_write = True
