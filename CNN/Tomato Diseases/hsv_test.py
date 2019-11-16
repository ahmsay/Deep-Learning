import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt

hsv_1_1 = (10, 5, 0)
hsv_1_2 = (70, 255, 255)

hsv_1_1_color = np.full((10, 10, 3), hsv_1_1, dtype = np.uint8) / 255.0
hsv_1_2_color = np.full((10, 10, 3), hsv_1_2, dtype = np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(hsv_1_1_color))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(hsv_1_2_color))
plt.show()