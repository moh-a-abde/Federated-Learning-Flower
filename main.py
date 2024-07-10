import flwr as fl
from client import generate_client_fn
from dataset import prepare_dataset
import pandas as pd
import yaml
import ray

MAX_RAM = 6 * 2**30  # 6 GiB

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

@ray.remote
def mean(*arrays):
    return np.vstack(arrays).mean(axis=0)

def get_res_parallel(simulations, num_loads):
    load_size = simulations.shape[0] // num_loads
    simulations_per_load = [simulations[round(n * load_size): round((n+1) * load_size)]
                            for n in range(num_loads)]
    result = mean.remote(*[ray.put(simulations) for simulations in simulations_per_load])
    return ray.get(result)

def get_expected_res(simulations, MAX_RAM=MAX_RAM):
    expected_result = np.zeros(shape=87_381, dtype=np.float64)
    bytes_per_res = len(expected_result) * (64 // 8)

    num_steps = simulations.shape[0] * bytes_per_res // MAX_RAM + 1
    step_size = simulations.shape[0] / num_steps

    print(f"Number of steps: {num_steps} ({step_size:,.0f} simulations each)")
    for n in range(num_steps):
        print(f"\r{n / num_steps:.0%}", end="")
        step_simulations = simulations[round(n * step_size): round((n+1) * step_size)]
        expected_result += get_res_parallel(simulations=step_simulations, num_loads=ray.available_resources()["CPU"])
        del step_simulations
    print(f"\r100%")

    return expected_result / num_steps

def main():
    config = load_config('conf/base.yaml')

    num_partitions = config['num_clients']
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    val_ratio = config.get('val_ratio', 0.1)
    csv_files = [config['data'][f'file_path_{i}'] for i in range(1, 6)]
    num_rounds = config['num_rounds']

    if len(csv_files) != num_partitions:
        raise ValueError(f"Number of CSV files ({len(csv_files)}) does not match number of clients ({num_partitions})")

    trainloaders, valloaders, datasets = prepare_dataset(
        num_partitions, batch_size, num_classes, val_ratio, csv_files
    )

    client_fn = generate_client_fn(trainloaders, valloaders, datasets[0])

    ray.init(num_cpus=4, memory=10 * 1024 * 1024 * 1024)  # Ensure this matches the available resources

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_partitions,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        resources_per_client={"num_cpus": 1, "memory": 2 * 1024 * 1024 * 1024}  # Allocate 1 CPU and 2 GB memory per client
    )

if __name__ == "__main__":
    main()
