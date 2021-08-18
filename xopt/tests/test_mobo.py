from xopt import Xopt

import yaml

# Make a proper input file.
YAML = """
xopt: {output_path: null, verbose: true}

algorithm:
  name: mobo
  options: {ref: [1.4, 1.4],
            n_initial_samples: 5,
            use_gpu: False,
            n_steps: 5, 
            generator_options: {batch_size: 2}
            verbose: True}

simulation: 
  name: test_TNK
  evaluate: xopt.evaluators.test_TNK.evaluate_TNK  

vocs:
  name: TNK_test
  description: null
  simulation: test_TNK
  templates: null
  variables:
    x1: [0, 3.14159]
    x2: [0, 3.14159]
  objectives: {y1: MINIMIZE, y2: MINIMIZE}
  constraints:
    c1: [GREATER_THAN, 0]
    c2: [LESS_THAN, 0.5]
  constants: {a: dummy_constant}

"""
config = yaml.safe_load(YAML)


def test_mobo_tnk():
    X = Xopt(config)

    X.random_evaluate()

    X.run()





