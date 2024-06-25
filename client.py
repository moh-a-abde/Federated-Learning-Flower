from collections import OrderedDict
from typing import Dict
from flwr.common import NDArrays, Scalar
import torch
import flwr as fl
import torch.optim as optim

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_classes, input_dim) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes, input_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)
    
    
    def get_parameters(self, config: Dict[str, Scalar]):
        
        return [ val.cpu().numpy() for _, val in self.model.state_dict().items()]

    
    def fit(self, parameters, config):

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']
        # Define the optimizer (e.g., Adam)
        
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        
        self.set_parameters(parameters)

        loss, accuarcy = test(self.model, self.valloader, self.device)
        
        return float(loss), len(self.valloader), {'accuarcy': accuarcy}
    


def generate_client_fn(trainloaders, valloaders, num_classes, input_dim):

    def client_fn(cid: str):

        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes,
                            input_dim=input_dim)


    return client_fn
