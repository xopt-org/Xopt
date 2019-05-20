import configparser
import os
from math import isnan
import numpy as np

"""
Code do deal with legacy decisions, constraits, objectives style files. 

"""


LegacyColumnNames = {'s_x':'x_rms', 's_y':'y_rms', 's_z':'z_rms', 's_dE':'deltaE_rms',
                             'ex':'x_normemit', 'ey':'y_normemit', 'ez':'z_normemit', 'KE': 'E_kinetic'}


VARIABLE_OPERATORS = ['VARY', '|->', 'LINK']
def parse_variables(file):
  """ Parses constraint, objective, and decision files and line of the form:

  decision:
  name VARY min   max
  str  str  float float

  returns variable, constant, and linked_variables dicts with:
  ( {name: (min, max)}, {name: (targetname, offset)}, {name: const} )

  """
  f = open(file)
  variables = {}
  constants = {}
  linked_variables = {}
  for line in f:
    s = line.split()
    if len(s) == 0: # Skip empty lines
      continue
    if s[0][0] == '#': # Skip commented lines
      continue
    name = s[0].split('[')[0] # Ignore brackets
    if name in LegacyColumnNames:     # legacy name support
      name = LegacyColumnNames[name]

    if s[1] in VARIABLE_OPERATORS:
      if s[1] == '|->' or s[1] == 'LINK':
        linked_variables.update({name: (s[2], float(s[4])) })
      else:
        min = float(s[2])
        max = float(s[3])
        if min == max:
          constants.update({name : min} )
        else:
          variables.update({name : (min, max )} )
  f.close()

  output = {'variables':variables, 'linked_variables':linked_variables, 'constants':constants }

  return output




CONSTRAINT_OPERATORS = ['GREATER_THAN', 'LESS_THAN']
def parse_constraints(file):
  """
  constraint:
  name OP  value
  str  str float
  where OP can be: GREATER_THAN, LESS_THAN

  returns dict with:
  {name: (OP, value)
  """
  f = open(file)
  constraints = {}
  for line in f:
    s = line.split()
    if len(s) == 0: # Skip empty lines
      continue
    if s[0][0] == '#': # Skip commented lines
      continue
    name = s[0].split('[')[0] # Ignore brackets
    if name in LegacyColumnNames:     # legacy name support
      name = LegacyColumnNames[name]
    if s[1] in CONSTRAINT_OPERATORS:
      constraints.update({name : (s[1], float(s[2]))} )
  f.close()
  return {'constraints':constraints}



OBJECTIVE_OPERATORS = ['MINIMIZE', 'MAXIMIZE']
OBJECTIVE_WEIGHTS = [-1.0, 1.0]
objective_weight = dict(zip(OBJECTIVE_OPERATORS, OBJECTIVE_WEIGHTS))
def parse_objectives(file):
  """
  constraint:
  name OP
  str  str
  where OP can be: [MINIMIZE, MAXIMIZE]

  returns dict with:
  {name: weight}
  """
  f = open(file)
  objectives = {}
  for line in f:
    s = line.split()
    if len(s) == 0: # Skip empty lines
      continue
    if s[0][0] == '#': # Skip commented lines
      continue
    name = s[0].split('[')[0] # Ignore brackets
    if name in LegacyColumnNames:     # legacy name support
      name = LegacyColumnNames[name]
    if s[1] in OBJECTIVE_OPERATORS:
      objectives.update({name : objective_weight[s[1]]} )
  f.close()
  return {'objectives':objectives}
  
  
  
  
  
def opt_params_from_opt_config(filePath):
  config = configparser.ConfigParser()

  config['DEFAULTS'] = {'objectives_file'      : 'objectives.in',
                        'constraints_file'     : 'constraints.in',
                        'variables_file'       : 'variables.in'}
  config.read(filePath)
  # OPT_CONFIG (cov files)
  opt_config = config['OPT_CONFIG']
  VAR_FILE = opt_config['variables_file']
  OBJ_FILE = opt_config['objectives_file']
  CON_FILE = opt_config['constraints_file']
  if not os.path.isabs(VAR_FILE):
    VAR_FILE = os.path.join(os.getcwd(), VAR_FILE)
  if not os.path.isabs(OBJ_FILE):
    OBJ_FILE = os.path.join(os.getcwd(), OBJ_FILE)
  if not os.path.isabs(CON_FILE):
    CON_FILE = os.path.join(os.getcwd(), CON_FILE)

  opt_params = {}
  opt_params.update(parse_variables(VAR_FILE))
  opt_params.update(parse_constraints(CON_FILE))
  opt_params.update(parse_objectives(OBJ_FILE))


  return opt_params  
    
  
def parse_docs(dir, dname='decisions', cname='constraints', oname='objectives', suffix=''):
    """
    Simple routine to parse docs files
    """
    d={}
    d.update(parse_constraints(os.path.join(dir, cname+suffix)))
    d.update(parse_objectives(os.path.join(dir, oname+suffix)))
    d.update(parse_variables(os.path.join(dir, dname+suffix)))
    return d  
  
  
# ------ Output evaluation--------

def evaluate_objectives(objective_dict, output):

  #print(objective_dict)
  #print(output)

  """
  Uses objective dict and output dict to return a list of objective values,
  ordered by sorting the objective keys.
  """
  obj = sorted_names(objective_dict)
  eval = []
  for o in obj:
    eval.append(output[o])
  return eval

def evaluate_objectives_inverse(objective_dict, eval):
  """
  inverse of evaluate_objectives, returns a dict
  """
  obj_names = sorted_names(objective_dict)
  output = {}
  for name, val in zip(obj_names, eval):
    output[name] = val
  return output

def evaluate_constraints(constraint_dict, output):
  """
  Use constraint dict and output dict to form a list of constraint evaluations.
  A constraint is satisfied if the evaluation is > 0.
  """
  con = sorted_names(constraint_dict)
  eval = []
  for c in con:
    x = output[c]
    op, d = constraint_dict[c]
    if isnan(x):
      # This is a NaN
      eval.append(-666.0)
    elif op   == 'GREATER_THAN':   # x > d -> x-d > 0
      eval.append( x-d )
    elif op == 'LESS_THAN':      # x < d -> d-x > 0
      eval.append( d-x )
  return eval
  
def constraint_satisfaction(constraint_dict, output):
    """
    Returns a dictionary of constraint names, and a bool with their satisfaction. 
    """
    vals = evaluate_constraints(constraint_dict, output)
    keys = sorted_names(constraint_dict)
    d = {}
    for k, v in zip(keys, vals):
        if v > 0:
            satisfied = True
        else:
            satisfied = False
        d[k] =  satisfied
    return d

def n_constraints_satistfied(constraint_dict, output):
    vals = np.array(evaluate_constraints(constraint_dict, output))
    return len(np.where(vals > 0)[0])

def evaluate_constraints_inverse(constraint_dict, eval):
  """
  inverse of evaluate_constraints, returns a dictionary
  """
  output = {}
  con_names = sorted_names(constraint_dict)
  for name, val in zip(con_names, eval):
    op, d = constraint_dict[name]
    if op == 'GREATER_THAN':
      output[name] = val + d
    elif op == 'LESS_THAN':
      output[name] = d - val
    else:
      print('ERROR: unknown constraint operator: ', op)
  return output



def toolbox_params(variable_dict={}, objective_dict={}, constraint_dict={}):
  """
  Returns a dict of parameters intended for a DEAP toolbox
  """
  params={} 
  params['WEIGHTS'] = tuple([objective_weight[objective_dict[k]] for k in sorted_names(objective_dict)])

  mins = var_mins(variable_dict)
  maxs = var_maxs(variable_dict)
  for low, up in zip(mins, maxs):
    if up < low:
      print('ERROR: bounds out of order: ', low, up)
      sys.exit()
  params['BOUND_LOW'] = mins
  params['BOUND_UP']  = maxs

  params['N_DIM'] = len(variable_dict)
  params['N_CONSTRAINTS'] = len(constraint_dict)

  # Labels for variables and objectives
  params['VARIABLE_LABELS'] = sorted_names(variable_dict)
  params['OBJECTIVE_LABELS'] = sorted_names(objective_dict)

  return params



def sorted_names(var_dict):
  names = list(var_dict.keys())
  names.sort()
  return names

def var_mins(var_dict):
  return [var_dict[name][0] for name in sorted_names(var_dict) ]

def var_maxs(var_dict):
  return [var_dict[name][1] for name in sorted_names(var_dict) ]
  
  
