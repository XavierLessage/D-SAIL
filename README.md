# D-SAIL - Distributed Secure AI Learning

Welcome to the D-SAIL's Github repository!

 [![Documentation Status](https://readthedocs.org/projects/secure-privacy-preserving-and-federated-machine-learning/badge/?version=latest)](https://secure-privacy-preserving-and-federated-machine-learning.readthedocs.io/en/latest/?badge=latest)

The documentation of the code is available on [readthedocs](https://secure-privacy-preserving-and-federated-machine-learning.readthedocs.io/en/latest/?badge=latest)

## Description

This repository currently hosts code to pseudonymized or anonymized medical images in the DICOM format and train models using PyTorch in a distributed fashion, using PyTorch. 

<!-- The general framework encompasses tools that allow the training of a global model from local and multiples models. The main advantage is that the participants does not exchange their local data while benefiting a ,presupposed, higher performing model. In exchange, they need to train a similar model on their local data. Particularly, in federated learning, a consortium of actors share the weights of their locally trained model and a central unit aggregate the latter. While allowing the partners to not share their data, the architecture rises several challenges in order to ensure a privacy-preserving system. First, the pseudonymization of the training dataset. Secondly, the confidentiality of the models and the gradients in order to prevent any reverse engineering to the training dataset. Eventually, the protection of the model against degradation by training on inadequate data. -->

## Quick start

0. Insertion metadata
1. Pseudonimization & Hashage
2. Dicom to img + json
3. classify data
4. cat to dataset
5. (hospital split)
6. federated learning

## References

The code to pseudonimize the DICOM files was adapted from https://github.com/KitwareMedical/dicom-anonymizer, please refer to their repository for details on initial implementation.
