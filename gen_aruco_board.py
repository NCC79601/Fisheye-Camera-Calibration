import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 设置 A4 纸尺寸（210 x 297 毫米）和 DPI
dpi = 300
a4_width_mm = 210
a4_height_mm = 297
a4_width = int(a4_width_mm / 25.4 * dpi)  # 将毫米转换为像素
a4_height = int(a4_height_mm / 25.4 * dpi)  # 将毫米转换为像素
pixels_per_mm = dpi / 25.4

# 已有的 ArUco board 参数
markersX = 4
markersY = 5
markerLength_mm = 35 # pixels
markerSeparation_mm = 15  # pixels
markerLength = int(markerLength_mm * pixels_per_mm)
markerSeparation = int(markerSeparation_mm * pixels_per_mm)
margins = 0
borderBits = 10

# 计算 ArUco board 的宽度和高度
width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins
height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins

# 获取预定义的字典并创建 ArUco board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.GridBoard((markersX, markersY), float(markerLength), float(markerSeparation), dictionary)
img = board.generateImage((width, height), margins, borderBits)

# 将 OpenCV 图像转换为 PIL 图像
img_pil = Image.fromarray(img)

# 创建一个白色背景的 A4 纸图像
a4_img = Image.new('RGB', (a4_width, a4_height), 'white')

# 计算 ArUco board 图像在 A4 纸上的位置（居中）
x_offset = (a4_width - img_pil.width) // 2
y_offset = (a4_height - img_pil.height) // 2

# 将 ArUco board 图像粘贴到 A4 纸图像上
a4_img.paste(img_pil, (x_offset, y_offset))

print(f'a4 image size: {a4_img.size}')

# 保存为 PDF
a4_img.save('aruco_board.pdf', 'PDF', resolution=dpi)