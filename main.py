import multiprocessing
import server
import client

if __name__ == "__main__":
    # Start server process
    server_process = multiprocessing.Process(target=server.start_server)
    server_process.start()

    # Start client processes
    clients = [multiprocessing.Process(target=client.main) for _ in range(5)]
    for c in clients:
        c.start()

    server_process.join()
    for c in clients:
        c.join()
