import numpy as np
import cv2
import matplotlib.pyplot as plt

# Checkerboard dimensions
checkerboard_size = (9, 6)  # 9x6 checkerboard
square_size = 25  # Size of a square in mm

# Create checkerboard
checkerboard = np.zeros(((checkerboard_size[1]+1)*square_size, (checkerboard_size[0]+1)*square_size), dtype=np.uint8)

# Fill squares
for i in range(checkerboard_size[1] + 1):
    for j in range(checkerboard_size[0] + 1):
        if (i+j) % 2 == 0:
            checkerboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255

# Display checkerboard
plt.imshow(checkerboard, cmap='gray')
plt.axis('off')  # Hide axes

# Save checkerboard as PDF
plt.savefig('./checkerboard.pdf', bbox_inches='tight', pad_inches=0)
plt.show()