import subprocess
import time

def start_server():
    server_process = subprocess.Popen(['python', 'server.py'])
    return server_process

def start_client():
    client_process = subprocess.Popen(['python', 'client.py'])
    return client_process

if __name__ == "__main__":
    # Start the server
    print("Starting server...")
    server_process = start_server()
    
    # Wait for the server to start
    time.sleep(5)

    # Start the clients
    num_clients = 3  # Adjust the number of clients as needed
    client_processes = []
    for i in range(num_clients):
        print(f"Starting client {i+1}...")
        client_process = start_client()
        client_processes.append(client_process)
    
    # Wait for all processes to finish
    server_process.wait()
    for client_process in client_processes:
        client_process.wait()
