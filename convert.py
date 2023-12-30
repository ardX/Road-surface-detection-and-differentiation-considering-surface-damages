import os
import glob
import cv2 as cv
import numpy as np
from pathlib import Path

path = Path('data/')

results_save = 'results'
path_rst = path/results_save
path_rst.mkdir(exist_ok=True)

colored_results = 'results_color'
path_crst = path/colored_results
path_crst.mkdir(exist_ok=True)

def colorfull(image):
  # grab the image dimensions
  #height = image.shape[0]
  #width = image.shape[1]
  width = 1080
  height = 1920
 
  # loop over the image, pixel by pixel
  for x in range(width):
    for y in range(height):
        b, g, r = frame[x, y]
        if (b, g, r) == (0,0,0): #background
            frame[x, y] = (0,0,0)
        elif (b, g, r) == (1,1,1): #roadAsphalt
            frame[x, y] = (85,85,255)
        elif (b, g, r) == (2,2,2): #roadPaved
            frame[x, y] = (85,170,127)
        elif (b, g, r) == (3,3,3): #roadUnpaved
            frame[x, y] = (255,170,127) 
        elif (b, g, r) == (4,4,4): #roadMarking
            frame[x, y] = (255,255,255) 
        elif (b, g, r) == (5,5,5): #speedBump
            frame[x, y] = (255,85,255)
        elif (b, g, r) == (6,6,6): #catsEye
            frame[x, y] = (255,255,127)          
        elif (b, g, r) == (7,7,7): #stormDrain
            frame[x, y] = (170,0,127) 
        elif (b, g, r) == (8,8,8): #manholeCover
            frame[x, y] = (0,255,255) 
        elif (b, g, r) == (9,9,9): #patchs
            frame[x, y] = (0,0,127) 
        elif (b, g, r) == (10,10,10): #waterPuddle
            frame[x, y] = (170,0,0)
        elif (b, g, r) == (11,11,11): #pothole
            frame[x, y] = (255,0,0)
        elif (b, g, r) == (12,12,12): #cracks
            frame[x, y] = (255,85,0)
 
  # return the colored image
  return image


fqtd = 0

filenames = [img for img in glob.glob(str(path_rst/"*.png"))]

filenames.sort()

for img in filenames:
  frame = cv.imread(img)
  frame =  colorfull(frame)
  frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
  img_split = f'{img}'
  img_split = img_split[14:]
  print(img_split)
  name = "%09d.png"%fqtd
  cv.imwrite(os.path.join(path_crst, img_split), frame)

  fqtd += 1
  print(fqtd)

print("Done!")