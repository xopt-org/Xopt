from xopt import Xopt
import pytest


class TestLegacy:
    legacy_yaml = """
xopt:
  output_path: .
  verbose: true
  algorithm: cnsga
  
algorithm:
  name: cnsga
  function: xopt.cnsga.cnsga
  options:
    max_generations: 1000
    population_size: 300
    crossover_probability: 0.9
    mutation_probability: 1.0 
    selection: auto
    verbose: true

simulation:
  name: impact_with_distgen
  evaluate: xopt.tests.evaluators.TNK.evaluate_TNK
  options:
    verbose: false
    workdir: .
    archive_path: .
    
vocs:
  name: LCLS cu_inj Impact-T and Disgten full optimization v9
  description: data set for 20 pc for lcls_cu_inj, 20k particles
  simulation: impact_with_distgen
  templates:
    impact_config: .
    distgen_input_file: .
  
  variables:

    # Distgen
    distgen:r_dist:sigma_xy:value: [0.1, 0.3]
    distgen:t_dist:length:value: [3.0, 12.0]
  linked_variables: null
  
  constants: null
    
  objectives:
    end_norm_emit_xy: MINIMIZE
  
  constraints:
    end_sigma_z: [LESS_THAN, 0.0015]    
"""

    def test_legacy(self):
        X = Xopt(self.legacy_yaml)
        print(X.config)
