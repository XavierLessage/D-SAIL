from opacus import PrivacyEngine, privacy_analysis
from fastcore.basics import *
from fastai.callback.all import *
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy


class DPCallback(Callback):
    def __init__(self, alphas, noise_multiplier, max_grad_norm, delta, device):
        store_attr()

    def before_step(self):
        for param in self.learn.model.parameters():
            param.accumulated_grads = []
        
        for x, y in zip(self.x, self.y):
            y_pred = self.learn.model(x[None])
            label = torch.tensor([y], device=device)
            new_loss = F.cross_entropy(y_pred, label)
            new_loss.backward()
            
            for param in self.learn.model.parameters():
                per_sample_grad = param.grad.detach().clone()
                clip_grad_norm_(per_sample_grad, max_norm=self.max_grad_norm)
                param.accumulated_grads.append(per_sample_grad)
                
        for param in self.learn.model.parameters():
            accumulated_grads = torch.stack(param.accumulated_grads, dim=0)
            param.grad = torch.mean(accumulated_grads, dim=0)
            
        for param in self.learn.model.parameters():
            noise = torch.normal(mean=0, std=self.noise_multiplier*self.max_grad_norm, size=param.size(), device=self.device)
            param.grad.data += noise

    def after_epoch(self):
        privacy_engine = PrivacyEngine(
            deepcopy(self.learn.model),
            sample_rate = self.learn.dls.bs/len(self.learn.dls.train_ds),
            alphas=self.alphas, 
            noise_multiplier=self.noise_multiplier, 
            max_grad_norm=self.max_grad_norm,
        )
        rdp = privacy_engine.get_renyi_divergence()
        epsilon, best_alpha = privacy_analysis.get_privacy_spent(self.alphas, rdp, self.delta)
        print(f"For sigma = {params['noise_multiplier']} => epsilon = {epsilon:.5f}, delta = {self.delta}")