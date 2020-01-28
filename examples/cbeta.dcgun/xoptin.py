

import json


if __name__== "__main__":
 
    data = {"constants": {"gun_voltage": 375},
            "constraints": {"end_t_rms": ["LESS_THAN",0.0015]},
            "description": "Test case",
            "name": "CBETA DC GUN",
            "objectives": {"end_norm_emitt_x": "MINIMIZE",
                           "end_qbunch": "MAXIMIZE"},
            "simulation": "gpt_with_distgen",
            "templates": {"distgen_input_file": "/Users/colwyngulliford/Documents/GitHub/xopt/examples/cbeta.dcgun/template/cbeta.dcgun.distgen.in.json",
                          "gpt_input_file": "/Users/colwyngulliford/Documents/GitHub/xopt/examples/cbeta.dcgun/template/cbeta.dcgun.gpt.in"},
            "variables": {"sol_1_current": [0,5],
                          "beam:params:sigma_xy:value": [0,5],
                          'beam:params:total_charge:value': [0,500]}
            }


    with open('cbeta.dcgun.vocs.json', 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
