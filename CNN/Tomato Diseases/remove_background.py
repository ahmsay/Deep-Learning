import cv2
import matplotlib.pyplot as plt
import glob

for filename in glob.glob('./test_samples/*.jpg'):

    sample = cv2.imread(filename)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    hsv_sample = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
    
    light_green = (35, 25, 0)
    dark_green = (103, 255, 255)
    
    light_yellow = (10, 25, 0)
    dark_yellow = (70, 255, 255)

    mask_green = cv2.inRange(hsv_sample, light_green, dark_green)
    result_green = cv2.bitwise_and(sample, sample, mask = mask_green)
    
    mask_yellow = cv2.inRange(hsv_sample, light_yellow, dark_yellow)
    result_yellow = cv2.bitwise_and(sample, sample, mask = mask_yellow)
    
    final_mask = mask_green + mask_yellow
    final_result = cv2.bitwise_and(sample, sample, mask = final_mask)
    
    plt.subplot(1, 2, 1)
    plt.imshow(sample)
    plt.subplot(1, 2, 2)
    plt.imshow(final_result)
    plt.show()