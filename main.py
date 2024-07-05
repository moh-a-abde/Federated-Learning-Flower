import multiprocessing
import server
import client

if __name__ == "__main__":
    # Start server process
    server_process = multiprocessing.Process(target=server.start_server)
    server_process.start()

    # Start client process
    client_process = multiprocessing.Process(target=client.main)
    client_process.start()

    server_process.join()
    client_process.join()
