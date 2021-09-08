import flwr as fl
from flwr.server.strategy import FedAvg
from fastai.vision.all import *

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np 

class SaveModelStrategy(fl.server.strategy.FedAvg):

    weight_path=None

    def aggregate_fit(
        self, 
        rnd:int, 
        results, 
        failures, 
    )-> Optional[fl.common.Weights]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            print(self.weight_path)
            if self.weight_path is not None:
                #Save weights
                print(f"Saving round {rnd} weights...")
                np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights

@call_parse
def main(
    savepath : Param("Set a path to save weights", str)=None
):

    strategy = SaveModelStrategy(
        fraction_fit = 1.0,
        min_fit_clients=2,
        min_available_clients = 2,
        min_eval_clients = 2
    )

    SaveModelStrategy.weight_path = savepath

    #fl.server.start_server(config={"num_rounds":1})
    fl.server.start_server(strategy=strategy, config={"num_rounds": 3})