from .cnsga import CNSGAGenerator
from .nsga2 import NSGA2Generator

registry = {"CNSGA": CNSGAGenerator, "NSGA2": NSGA2Generator}
