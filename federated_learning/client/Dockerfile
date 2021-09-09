FROM ubuntu:20.04

# Dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git curl

# Install flower and dependencies for machine learning
RUN python3 --version
RUN pip3 install flwr==0.16.0 numpy==1.21.2 fastai==2.5.2 torchvision==0.10.0 matplotlib==3.4.3 scikit-learn==0.24.2 opacus==0.14.0

EXPOSE 8081
COPY . .
CMD python3 client.py 