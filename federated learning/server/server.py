import flwr as fl
from flwr.server.strategy import FedAvg

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np 

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self, 
        rnd:int, 
        results, 
        failures, 
    )-> Optional[fl.common.Weights]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            #Save weights
            print(f"Saving round {rnd} weights...")
            np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights

if __name__ == '__main__':

    
    strategy = SaveModelStrategy(
        fraction_fit = 1.0,
        min_fit_clients=2,
        min_available_clients = 2,
        min_eval_clients = 2
    )
    


    #fl.server.start_server(config={"num_rounds":1})
    fl.server.start_server(strategy=strategy, config={"num_rounds": 3})