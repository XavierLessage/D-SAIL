import flwr as fl
from fastcore.script import *
import numpy as np
import os 

class SaveModelStrategy(fl.server.strategy.FedAvg):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = None

    def aggregate_fit(
        self, 
        rnd:int, 
        results, 
        failures, 
    ):
        weights = super().aggregate_fit(rnd, results, failures)
        print(self.save_path)
        if (weights is not None) and (self.save_path is not None):
            print(f"Saving round {rnd} weights...")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path) 
            np.savez(self.save_path+"/round-"+str(rnd)+"-weights.npz", *weights)
        return weights


@call_parse
def main(
    fraction_fit : Param("The fraction of available client used for training", float)=1.0,
    min_fit_clients : Param("The minimum number of clients used to start training", int)=3,
    min_available_clients : Param("The minimum number of clients used to start server", int)=3,
    min_eval_clients : Param("The minimum number of clients used for evaluation", int)=3,
    num_rounds : Param("The number of rounds of training", int)=3,
    save_path : Param("Set a path to save weights", str)=None
):

    strategy = SaveModelStrategy(
        fraction_fit = fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients = min_available_clients,
        min_eval_clients = min_eval_clients
    )

    if save_path is not None: strategy.save_path = save_path
    

    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})