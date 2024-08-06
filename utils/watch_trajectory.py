import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

# 读取 pickle 文件
curdir = os.path.normpath(os.path.dirname(__file__))
file_path = os.path.join(os.path.dirname(curdir), 'saved_trajectory/2024-08-05_12-20-35.pkl')
print(f'file path: {file_path}')

with open(file_path, 'rb') as file:
    trajectory_data = pickle.load(file)
print('trajecotry data loaded.')

# Extracting trajectory data
t = [data['t'] for data in trajectory_data]
x = [data['pose'][0] for data in trajectory_data]
y = [data['pose'][1] for data in trajectory_data]
z = [data['pose'][2] for data in trajectory_data]

min_val = min(min(x), min(y), min(z))
max_val = max(max(x), max(y), max(z))

# Plotting trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])
ax.set_zlim([min_val, max_val])
ax.set_title('Trajectory')

# Animating trajectory
def update(frame):
    ax.cla()
    ax.plot(x[:frame], y[:frame], z[:frame])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])
    ax.set_title('Trajectory')
    
ani = FuncAnimation(fig, update, frames=len(t), interval=100)
plt.show()