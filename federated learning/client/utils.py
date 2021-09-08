import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc 

def set_seed(dls, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    dls.rng.seed(seed)

def save_matrix(learn, path):
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(7,7))
    plt.savefig(path)


def save_roc(learn, path):
    preds, y, loss = learn.get_preds(with_loss=True)
    probs = np.exp(preds[:, 1])
    fpr, tpr, _ = roc_curve(y, probs, pos_label=1)
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
    plt.savefig(path)