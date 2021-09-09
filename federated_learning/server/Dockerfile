FROM ubuntu:20.04

# Dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git curl

# Install flower and dependencies for machine learning
RUN python3 --version
RUN pip3 install flwr==0.16.0 numpy==1.21.2 fastcore==1.3.26

EXPOSE 8080

COPY . .
CMD python3 server.py 