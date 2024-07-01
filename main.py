import pickle
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl
from pprint import pprint

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from model import train_xgboost, Net, train_nn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    # 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    trainloaders, validationloaders, testloader, input_dim = prepare_dataset(cfg.num_clients, 
                                                                  cfg.batch_size, num_classes=cfg.num_classes)
    # Check trainloaders, validationloaders, testloader
    print(f"Train loaders: {len(trainloaders)}, Validation loaders: {len(validationloaders)}, Test loader: {len(testloader)}, Input dim: {input_dim}'")
    
    # 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, testloader, cfg.num_classes, input_dim, cfg.model_types)
    
    # 4. Define your strategy
    def evaluate_metrics_aggregation_fn(metrics):
        accuracies = [metric[1]['accuracy'] for metric in metrics]
        aggregated_accuracy = sum(accuracies) / len(accuracies)
        return {'accuracy': aggregated_accuracy}
    
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, input_dim, testloader),
                                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)

    # 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 4},
    )

    # 6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    # 7. Train XGBoost classifier if specified in config
    if 'xgb' in cfg.model_types:
        xgboost_model = train_xgboost()

    # Save models and history as required
    results = {
        'history': history,
        'xgboost_model': xgboost_model if 'xgb' in cfg.model_types else None
    }

    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
