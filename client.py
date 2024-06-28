from collections import OrderedDict
from typing import Dict
import flwr as fl
from flwr.common import NDArrays, Scalar

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, input_dim) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes, input_dim)

    def set_parameters(self, parameters):
        # Save the parameters to a temporary file and load them into the model
        temp_model_path = 'temp_model.json'
        with open(temp_model_path, 'wb') as f:
            f.write(parameters[0])
        self.model.load_model(temp_model_path)
    
    def get_parameters(self, config: Dict[str, Scalar]):
        # Save the model parameters to a temporary file and return the file contents
        temp_model_path = 'temp_model.json'
        self.model.save_model(temp_model_path)
        with open(temp_model_path, 'rb') as f:
            parameters = [f.read()]
        return parameters
    
    def fit(self, parameters, config):
        # Copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        
        # No need to define optimizer for XGBoost

        # Do local training
        train(self.model, self.trainloader, None, epochs, "cpu")

        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, "cpu")
        
        return float(loss), len(self.valloader), {'accuracy': accuracy}

def generate_client_fn(trainloaders, valloaders, num_classes, input_dim):
    def client_fn(cid: str):
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,
                            input_dim=input_dim)
    return client_fn
