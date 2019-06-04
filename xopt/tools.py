

import numpy as np
import json



xopt_logo = """                 _   
                | |  
__  _____  _ __ | |_ 
\ \/ / _ \| '_ \| __|
 >  < (_) | |_) | |_ 
/_/\_\___/| .__/ \__|
          | |        
          |_|        
"""


#--------------------------------
# VOCS utilities
def save_vocs(vocs_dict, filePath=None):
    """
    Write VOCS dictionary to a JSON file. 
    If no filePath is given, the name is chosen from the 'name' key + '.json'
    """

    if filePath:
        name = filePath
    else:
        name = vocs_dict['name']+'.json'
    with open(name, 'w') as outfile:
        json.dump(vocs_dict, outfile, ensure_ascii=True, indent='  ')
    print(name, 'written')
    
    
def load_vocs(filePath):
    """
    Load VOCS from a JSON file
    Returns a dict
    """
    with open(filePath, 'r') as f:
        dat = json.load(f)
    return dat    
    


def random_settings(vocs, include_constants=False):
    """
    Uniform sampling of the variables described in vocs['variables'] = min, max.
    Returns a dict of settings. 
    If include_constants, the vocs['constants'] are added to the dict. 
    
    """
    settings = {}
    for key, val in vocs['variables'].items():
        a, b = val
        x = np.random.random()
        settings[key] = x*a + (1-x)*b
    if include_constants:
        for key, val in vocs['constants'].items():
            settings[key] = val
    return settings    



#--------------------------------
# Vector encoding and decoding
    
# Decode vector to dict
def decode1(vec, labels):
    return dict(zip(labels, vec.tolist()))
# encode dict to vector
def encode1(d, labels):
    return [d[key] for key in labels]    