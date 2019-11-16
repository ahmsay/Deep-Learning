import os
import cv2
import glob
from PIL import Image

light_green = (35, 25, 0)
dark_green = (103, 255, 255) # 120, 255, 255

light_yellow = (10, 25, 0) # 10, 50, 0
dark_yellow = (70, 255, 255)

directories = []

for index, (root, dirs, files) in zip(range(1), os.walk('./dataset')):
    directories = dirs

for directory in directories:
    for filename in glob.glob('./dataset/' + directory + '/*.jpg'):
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        hsv_sample = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        mask_green = cv2.inRange(hsv_sample, light_green, dark_green)
        result_green = cv2.bitwise_and(im, im, mask = mask_green)
        
        mask_yellow = cv2.inRange(hsv_sample, light_yellow, dark_yellow)
        result_yellow = cv2.bitwise_and(im, im, mask = mask_yellow)
        
        final_mask = mask_green + mask_yellow
        final_result = cv2.bitwise_and(im, im, mask = final_mask)
        
        filename = filename.split('\\')[1]
        img = Image.fromarray(final_result, 'RGB')
        img.save('dataset_processed/' + directory + '/' + filename)