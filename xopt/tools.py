import numpy as np

# for dummy executor
from concurrent.futures import Future, Executor
from threading import Lock

from datetime import date
from hashlib import blake2b
import yaml
import json
from copy import deepcopy
import importlib
import inspect
import datetime
import os
import logging

xopt_logo = """  _   
                | |  
__  _____  _ __ | |_ 
\ \/ / _ \| '_ \| __|
 >  < (_) | |_) | |_ 
/_/\_\___/| .__/ \__|
          | |        
          |_|        
"""

"""UTC to ISO 8601 with Local TimeZone information without microsecond"""


def isotime():
    return datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).astimezone().replace(
        microsecond=0).isoformat()


logger = logging.getLogger(__name__)


# --------------------------------
# Config utilities

def load_config(source):
    """
    Returns a dict loaded from a JSON or YAML file, or string. 
    
    If source is already a dict, just returns the same dict.
    
    """
    if isinstance(source, dict):
        logger.info('Loading config from dict.')
        return source

    if isinstance(source, str):
        if os.path.exists(source):
            if source.endswith('.json'):
                logger.info(f'Loading from JSON file: {source}')
                return json.load(open(source))
            elif source.endswith('.yaml'):
                logger.info(f'Loading from YAML file: {source}')
                return yaml.safe_load(open(source))
            else:
                logger.error(f'Cannot load file {source}')
        else:
            logger.info('Loading config from text')
            return yaml.safe_load(source)
    else:
        raise Exception(f'Do not know how to load {source}')


def save_config(data, filename, verbose=True):
    """
    Saves data to a JSON or YAML file, chosen by the filename extension.  
    
    """
    if filename.endswith('json'):
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=True, indent='  ',
                     cls=NpEncoder)
        if verbose:
            logger.info(f'Config written as JSON to {filename}')
    elif filename.endswith('yaml'):
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)
        if verbose:
            logger.info(f'Config written as YAML to {filename}')
    else:
        raise


# --------------------------------
# VOCS utilities
def save_vocs(vocs_dict, filePath=None):
    """
    Write VOCS dictionary to a JSON file. 
    If no filePath is given, the name is chosen from the 'name' key + '.json'
    """

    if filePath:
        name = filePath
    else:
        name = vocs_dict['name'] + '.json'
    with open(name, 'w') as outfile:
        json.dump(vocs_dict, outfile, ensure_ascii=True, indent='  ')
    logger.info(name, 'written')


def load_vocs(filePath):
    """
    Load VOCS from a JSON file
    Returns a dict
    """
    with open(filePath, 'r') as f:
        dat = json.load(f)
    return dat


def random_settings(vocs, include_constants=True, include_linked_variables=True):
    """
    Uniform sampling of the variables described in vocs.variables = min, max.
    Returns a dict of settings. 
    If include_constants, the vocs.constants are added to the dict. 
    
    """
    settings = {}
    for key, val in vocs.variables.items():
        a, b = val
        x = np.random.random()
        settings[key] = x * a + (1 - x) * b

    # Constants    
    if include_constants and vocs.constants:
        settings.update(vocs.constants)

    # Handle linked variables
    if include_linked_variables and 'linked_variables' in vocs and vocs[
        'linked_variables']:
        for k, v in vocs['linked_variables'].items():
            settings[k] = settings[v]

    return settings


def random_settings_arrays(vocs, n, include_constants=True,
                           include_linked_variables=True):
    """
    Similar to random_settings, but with arrays of size n. 
    
    Uniform sampling of the variables described in vocs.variables = min, max.
    Returns a dict of settings, with each settings as an array. 

    If include_constants, the vocs.constants are added to the dict as full arrays.
    
    """
    settings = {}
    for key, val in vocs.variables.items():
        a, b = val
        x = np.random.random(n)
        settings[key] = x * a + (1 - x) * b

    # Constants    
    if include_constants and 'constants' in vocs and vocs.constants:
        for k, v in vocs.constants.items():
            settings[k] = np.full(n, v)

    # Handle linked variables
    if include_linked_variables and 'linked_variables' in vocs and vocs[
        'linked_variables']:
        for k, v in vocs['linked_variables'].items():
            settings[k] = np.full(n, settings[v])

    return settings


# --------------------------------
# Vector encoding and decoding

# Decode vector to dict
def decode1(vec, labels):
    return dict(zip(labels, vec.tolist()))


# encode dict to vector
def encode1(d, labels):
    return [d[key] for key in labels]


# --------------------------------
# Paths

def full_path(path, ensure_exists=True):
    """
    Makes path abolute. Can ensure exists. 
    """
    p = os.path.expandvars(path)
    p = os.path.abspath(p)
    if ensure_exists:
        assert os.path.exists(p), 'path does not exist: ' + p
    return p


def add_to_path(path, prepend=True):
    """
    Add path to $PATH
    """
    p = full_path(path)

    if prepend:
        os.environ['PATH'] = p + os.pathsep + os.environ['PATH']
    else:
        # just append
        os.environ['PATH'] += os.pathsep + p
    return p


def expand_paths(nested_dict, suffixes=['_file', '_path', '_bin'], verbose=True,
                 sep=' : ', ensure_exists=False):
    """
    Crawls through a nested dict and expands the path of any key that ends 
    with characters in the suffixes list. 

    Internally flattens, and unflattens a dict to this using a seperator string sep
    
    """
    d = flatten_dict(nested_dict, sep=sep)

    for k, v in d.items():
        k2 = k.split(sep)
        if len(k2) == 1:
            k2 = k2[0]
        else:
            k2 = k2[-1]

        if any([k2.endswith(x) for x in suffixes]):
            if not v:
                if verbose:
                    logger.warning(f'Warning: No path set for key {k}')
                continue
            if not isinstance(v, str):
                # Not a path
                continue

            file = full_path(v, ensure_exists=ensure_exists)
            if os.path.exists(file):
                d[k] = file
            else:
                if verbose:
                    logger.warning(f'Warning: Path {v} does not exist for key {k}')

    return unflatten_dict(d, sep=sep)


# --------------------------------
# filenames
def new_date_filename(prefix='', suffix='.json', path=''):
    """
    Gets a filename that doesn't exist based on the date
    
    
    Example: 
        new_date_filename('sample-', '.json', '.')
    Returns:
        './sample-2020-02-09-1.json'
    
    """
    counter = 1
    while True:
        name = f'{prefix}{date.today()}-{counter}{suffix}'
        file = os.path.join(path, name)
        if os.path.exists(file):
            counter += 1
        else:
            break
    return file


# --------------------------------
# h5 utils

def write_attrs(h5, group_name, data):
    """
    Simple function to write dict data to attribues in a group with name
    """
    g = h5.create_group(group_name)
    for key in data:
        g.attrs[key] = data[key]
    return g


def write_attrs_nested(h5, name, data):
    """
    Recursive routine to write nested dicts to attributes in a group with name 'name'
    """
    if type(data) == dict:
        g = h5.create_group(name)
        for k, v in data.items():
            write_attrs_nested(g, k, v)
    else:
        h5.attrs[name] = data

    # --------------------------------


# data fingerprinting
class NpEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "dict"): # Works with PyDantic models
            return obj.dict()                
        else:
            return super(NpEncoder, self).default(obj)


def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data. 
    Used JSON dumps to form strings, and the blake2b algorithm to hash.
    
    """
    h = blake2b(digest_size=16)
    for key in keyed_data:
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()


# --------------------------------
# nested dict flattening, unflattening

def flatten_dict(dd, sep=':', prefix=''):
    """
    Flattens a nested dict into a single dict, with keys concatenated with sep.
    
    Similar to pandas.io.json.json_normalize
    
    Example:
        A dict of dicts:
            dd = {'a':{'x':1}, 'b':{'d':{'y':3}}}
            flatten_dict(dd, prefix='Z')
        Returns: {'Z:a:x': 1, 'Z:b:d:y': 3}
    
    """
    return {prefix + sep + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, sep, kk).items()
            } if (dd and isinstance(dd, dict)) else {prefix: dd}  # handles empty dicts


def unflatten_dict(d, sep=':', prefix=''):
    """
    Inverse of flatten_dict. Forms a nested dict.
    """
    dd = {}
    for kk, vv in d.items():
        if kk.startswith(prefix + sep):
            kk = kk[len(prefix + sep):]

        klist = kk.split(sep)
        d1 = dd
        for k in klist[0:-1]:
            if k not in d1:
                d1[k] = {}
            d1 = d1[k]

        d1[klist[-1]] = vv
    return dd


def update_nested_dict(d, settings, verbose=False):
    """
    Updates a nested dict with flattened settings
    """
    flat_params = flatten_dict(d)

    for key, value in settings.items():
        if verbose:
            if key in flat_params:
                logger.info(f'Replacing param {key} with value {value}')
            else:
                logger.info(f'New param {key} with value {value}')
        flat_params[key] = value

    new_dict = unflatten_dict(flat_params)

    return new_dict


# --------------------------------
# Function manipulation


def get_function(name):
    """
    Returns a function from a fully qualified name or global name.
    """

    # Check if already a function
    if callable(name):
        return name

    if not isinstance(name, str):
        raise ValueError(f'{name} must be callable or a string.')

    if name in globals():
        if callable(globals()[name]):
            f = globals()[name]
        else:
            raise ValueError(f'global {name} is not callable')
    else:
        if '.' in name:
            # try to import
            m_name, f_name = name.rsplit('.', 1)
            module = importlib.import_module(m_name)
            f = getattr(module, f_name)
        else:
            raise Exception(f'function {name} does not exist')

    return f


def get_function_defaults(f):
    """
    Returns a dict of the non-empty POSITIONAL_OR_KEYWORD arguments.
    
    See the `inspect` documentation for detauls.
    """
    defaults = {}
    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # print(k, v.default, v.kind)
            if v.default != inspect.Parameter.empty:
                defaults[k] = v.default
    return defaults


def get_n_required_fuction_arguments(f):
    """
    Counts the number of required function arguments using the `inspect` module.
    """
    n = 0
    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if v.default == inspect.Parameter.empty:
                n += 1
    return n


# Dummy executor

class DummyExecutor(Executor):
    """
    Dummy executor. 
    
    From: https://stackoverflow.com/questions/10434593/dummyexecutor-for-pythons-futures
    
    """

    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
