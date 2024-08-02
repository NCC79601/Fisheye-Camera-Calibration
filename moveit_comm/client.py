import numpy as np
import socket
import time

server_ip = "127.0.0.1"
server_port = 60123
# TODO: move configs to json file

class ClientSocket():
    def __init__(self) -> None:
        self.host = server_ip
        self.port = server_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(30)
        print(f"Connecting to server at {self.host}:{self.port}...")
        self.client.connect((self.host, self.port))
        print("Connected to server.")

        
    def cli_send(self, pose):
        # pose is a 6darray with type float64
        # pose = np.array([0.3, 0.3, 0.3, 0, 0, 0, 1], dtype=np.float64)
        pose = np.array(pose, dtype=np.float64).reshape(-1)
        # print(f'pose shape: {pose.shape}')
        pose = pose.tobytes()
        self.client.sendall(pose)
        # time.sleep(0.1)
