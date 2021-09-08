import flwr as fl
from fastcore.basics import *
from dsail.utils import *
from dsail.differential_privacy import *

from collections import OrderedDict


class FLClient():
    def __init__(self, learn, lr, ep, apply_dp, alphas, noise_multiplier, max_grad_norm, delta, device, csv_path, data_path, matrix_path, roc_path):
        store_attr()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.learn.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.learn.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        cbs = []
        if self.apply_dp: cbs.append(DPCallback(alphas=self.alphas, noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm,delta = self.delta, device=self.device))
        if self.csv_path is not None:  cbs.append(CSVLogger(fname=self.csv_path+str(self.noise_multiplier)+'_'+str(self.data_path.name)))
        
        self.learn.fine_tune(self.ep, self.lr, cbs=cbs)

        return self.get_parameters(), len(self.learn.dls.train), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        res = self.learn.validate()
        loss, accuracy = res[0], res[1]

        if self.matrix_path is not None: save_matrix(self.learn, self.matrix_path)
        if self.roc_path is not None: save_roc(self.learn, self.roc_path)

        return float(loss), len(self.learn.dls.valid), {"accuracy":float(accuracy)}