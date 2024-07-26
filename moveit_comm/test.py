import socket
import numpy as np

server_ip = "127.0.0.1"
server_port = 60123

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((server_ip, server_port))
server.settimeout(30)  # Set a timeout of 10 seconds
server.listen(1)
print(f"Server listening on {server_ip}:{server_port}")
conn, addr = server.accept()
print(f"Connected by {addr}")

while True:
    data = conn.recv(1024)  # Adjust buffer size if necessary
    if not len(data):
        continue
    pose = np.frombuffer(data, dtype=np.float64)
    print(f"Received pose: {pose}")

conn.close()  # Close the connection