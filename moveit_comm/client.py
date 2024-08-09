import numpy as np
import zerorpc
import json

# server_ip = "127.0.0.1"
# server_port = 60123

import os
config_file_path = os.path.join(os.path.dirname(__file__), 'comm_config.json')

with open(config_file_path, 'r') as f:
    server_config = json.load(f)

server_ip = server_config['server_ip']
server_port = server_config['server_port']

class ClientSocket():
    def __init__(self) -> None:
        self.host = server_ip
        self.port = server_port
        self.client = zerorpc.Client()
        print(f"Connecting to server at {self.host}:{self.port}...")
        self.client.connect(f"tcp://{self.host}:{self.port}")
        print("Connected to server.")

        
    def cli_send(self, pose):
        # pose is a 6darray with type float64
        # pose = np.array([0.3, 0.3, 0.3, 0, 0, 0, 1], dtype=np.float64)
        pose = np.array(pose, dtype=np.float64).reshape(-1)
        # print(f'pose shape: {pose.shape}')
        pose = pose.tolist()
        self.client.command(pose)
        # time.sleep(0.1)
