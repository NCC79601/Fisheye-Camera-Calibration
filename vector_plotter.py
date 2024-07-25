import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class VectorPlotter:
    def __init__(self, vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], origin = [0, 0, 0]):
        self.vector_list = []
        for i in range(len(vectors)):
            self.vector_list.append(vectors[i])
        # self.vector = np.array(vector)
        self.origin = np.array(origin)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=30, azim=-60)
        self.draw_vectors()
        plt.ion()
        plt.show()

    def draw_vectors(self):
        plt.clf()  # 清除当前图形
        fig = plt.gcf()  # 获取当前图形
        # 检查是否已经有轴存在，并获取当前的视角
        if hasattr(self, 'ax'):
            elev, azim = self.ax.elev, self.ax.azim
        else:
            elev, azim = None, None
        self.ax = fig.add_subplot(111, projection='3d')  # 创建一个新的3D轴
        # 如果之前的视角存在，则应用它们
        if elev is not None and azim is not None:
            self.ax.view_init(elev=elev, azim=azim)
        color_list = ['r', 'g', 'b', 'c', 'm', 'y']
        for i in range(len(self.vector_list)):
            vector = self.vector_list[i]
            self.ax.quiver(
                self.origin[0],
                self.origin[1],
                self.origin[2],
                vector[0],
                vector[1],
                vector[2],
                color=color_list[i % 6]
            )  # 绘制向量
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        # plt.ioff()

    def update_vectors(self, new_vectors, new_origin = None):
        if new_origin is not None:
            self.origin = np.array(new_origin)
        self.vector_list = new_vectors
        self.draw_vectors()


if __name__ == '__main__':
    # Example usage
    vector = [0.1, 0.1, 0.1]
    dir = [1, 1, 1]

    plotter = VectorPlotter(vector)
    
    while True:
        vector = np.array(vector) + np.array(dir) * 0.01
        plotter.update_vector(vector)
        plt.pause(0.2)