#Import fastai
from fastai.data.all import *
from fastai.vision.all import *
from fastai.callback.all import *

#Import torch utils
from torch.nn.utils import clip_grad_norm_
import torchvision.models as models

#Import signal processing libs 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc 
import numpy as np 

#Import DP lib
from opacus.utils import module_modification
from opacus import PrivacyEngine
from opacus import privacy_analysis

#Import federated learning lib
import flwr as fl

from copy import copy 
from copy import deepcopy
import os

#resnet18
MODEL = models.resnet18(pretrained=True)
MODEL.fc = nn.Linear(512, 2)

SAVE = True #save roc curve and confusion matrix if True
DP = True #apply dp if true
SAVE_CSV = True #save training history if true

DEVICE='cuda:0' 

#set manual seed 
def set_seed(dls, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    dls.rng.seed(seed)

#Diff Privacy Callback for fastai
class DPCallback_Simple(Callback):
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

#FL client for federated learning
class FLClient(fl.client.NumPyClient):
    def __init__(self, learn, lr, ep):
        store_attr()
        self.learn = learn
        self.lr = lr
        self.ep = ep

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.learn.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.learn.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.learn.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        dp = DPCallback_Simple(params['alphas'], params['noise_multiplier'], params['max_grad_norm'], params['delta'], params['device'])
        if DP:
            if SAVE_CSV:  
                self.learn.fit_one_cycle(params['epochs'], lr_max=params['lr'], cbs=[dp,CSVLogger(fname=CSV_PTH, append=True)])
            else:
                self.learn.fit_one_cycle(params['epochs'], lr_max=params['lr'], cbs=[dp])
        else:
            self.learn.fit_one_cycle(params['epochs'])
        return self.get_parameters(), len(self.learn.dls.train), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        res = self.learn.validate()
        loss, accuracy = res[0], res[1]

        #Compute confmat from training
        interp = ClassificationInterpretation.from_learner(self.learn)
        losses, idxs = interp.top_losses()
        interp.plot_confusion_matrix(figsize=(7,7))

        if SAVE:
            print('Saving confusion matrix...')
            hospital = os.path.basename(__file__).replace('.', "")
            s = str(params['noise_multiplier']).replace('.', "")
            plt.savefig(f"./confusion matrix/{hospital}_{MODEL.__class__.__name__}_sig{s}")
        
        #Compute roc 
        preds, y, _loss = self.learn.get_preds(with_loss=True)
        probs = np.exp(preds[:, 1])
        fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)
        #Compute roc area
        roc_auc = auc(fpr, tpr)
        print(f'ROC area is {roc_auc}')
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Worst case')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        if SAVE:
            print('Saving roc curve...')
            s = str(params['noise_multiplier']).replace('.', "")
            hospital = os.path.basename(__file__).replace('.', "")
            plt.savefig(f"./roc/{hospital}_{MODEL.__class__.__name__}_sig{s}")

        return float(loss), len(self.learn.dls.valid), {"accuracy":float(accuracy)}


if __name__ == '__main__':

    #set torch device
    device=torch.device(DEVICE)

    datapath = "../Hospitals" 
    fnames = get_image_files(datapath+"/H0") #define the datapath of your organisation

    #Create datablock (class from fastai defining behavior of dataloaders)
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
        get_items = get_image_files,
        get_y = parent_label,
        splitter = GrandparentSplitter())

    dls = dblock.dataloaders(datapath+"/H0", bs=32, num_workers=0)

    set_seed(dls, 42)
    
    MODEL = module_modification.convert_batchnorm_modules(MODEL) #Batchnorm->GroupNorm to keep DP
    learn = Learner(dls, MODEL, metrics=[accuracy, RocAucBinary()])

    #params
    params = {'alphas': range(2,32),
        'noise_multiplier': 0.5,
        'max_grad_norm': 1.0,
        'delta':1e-5,
        'device': device, 
        'lr' : 3e-3 / 5,
        'epochs' : 2
    }

    if SAVE_CSV:
        s = str(params['noise_multiplier']).replace('.', "")
        hospital = os.path.basename(__file__).replace('.', "")
        CSV_PTH = f"{hospital}_{MODEL.__class__.__name__}_sig{s}.csv"
    #Training loop
    '''
    if DP:
        if SAVE_CSV:  
            learn.fit_one_cycle(NUM_EPOCHS, lr_max=3e-3 / 5, cbs=[DPCallback_Simple(alphas=range(2,32), noise_multiplier=NOISE_FACTOR, max_grad_norm=1.0,delta = DELTA, device=device), CSVLogger(fname=CSV_PTH)])
        else:
            learn.fit_one_cycle(NUM_EPOCHS, lr_max=3e-3 / 5, cbs=DPCallback_Simple(alphas=range(2,32), noise_multiplier=NOISE_FACTOR, max_grad_norm=1.0, delta=DELTA, device=device))
    else:
        learn.fit_one_cycle(NUM_EPOCHS)
    
    
    #Compute confmat from training
    inter = ClassificationInterpretation.from_learner(learn)
    losses, idxs = inter.top_losses()
    interp.plot_confusion_matrix(figsize=(7,7))

    if SAVE:
        plt.savefig(f"../COVID-19_Radiography_Dataset/confusion matrix/{MODEL_STR}")
    
    #Compute roc 
    preds, y, loss = learn.get_preds(with_loss=True)
    probs = np.exp(preds[:, 1])
    fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)
    #Compute roc area
    roc_auc = auc(fpr, tpr)
    print(f'ROC area is {roc_auc}')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Worst case')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    if SAVE:
        plt.savefig(f"COVID-19_Radiography_Dataset/roc/{MODEL_STR}")

    '''
    fl.client.start_numpy_client("localhost:8080", client=FLClient(learn, params['lr'], params['epochs']))