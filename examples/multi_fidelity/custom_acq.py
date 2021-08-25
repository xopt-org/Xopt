from botorch.acquisition.analytic import UpperConfidenceBound

def acq(model):
    return UpperConfidenceBound(model, 0.01)
