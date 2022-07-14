import cv2
import numpy as np

img = cv2.imread('example.jpeg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bound_lower = np.array([0, 100, 45])
bound_upper = np.array([225, 250, 255])
maskOrange = cv2.inRange(hsv_img, bound_lower, bound_upper)
maskingData = np.invert(maskOrange)

white_pt_coords=np.argwhere(maskingData)
min_y = min(white_pt_coords[:,0])
min_x = min(white_pt_coords[:,1])
max_y = max(white_pt_coords[:,0])
max_x = max(white_pt_coords[:,1])

crop = img[min_y:max_y,min_x:max_x]
resized = cv2.resize(crop, (576,216), interpolation = cv2.INTER_AREA)
print(np.average(hsv_img[0]))
# print(hsv_img[1])
# cv2.imshow('Crop',resized)



# cv2.waitKey(0)
# cv2.destroyAllWindows()