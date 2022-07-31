import cv2
import numpy as np

img = cv2.imread('example2.png')
# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# bound_lower = np.array([0, 100, 45])
# bound_upper = np.array([225, 250, 255])
# maskOrange = cv2.inRange(hsv_img, bound_lower, bound_upper)
# maskingData = np.invert(maskOrange)

im_gray = cv2.imread('example2.png', cv2.IMREAD_GRAYSCALE)
(thresh, maskingData) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

white_pt_coords=np.argwhere(maskingData)
min_y = min(white_pt_coords[:,0])
min_x = min(white_pt_coords[:,1])
max_y = max(white_pt_coords[:,0])
max_x = max(white_pt_coords[:,1])

crop = img[min_y:max_y,min_x:max_x]
resized = cv2.resize(crop, (576,216), interpolation = cv2.INTER_AREA)
# print(np.average(hsv_img[0]))


# print(hsv_img[1])
cv2.imshow('Crop',maskingData)



cv2.waitKey(0)
cv2.destroyAllWindows()

# im_gray = cv2.imread('ttd.jpeg', cv2.IMREAD_GRAYSCALE)
# (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# thresh = 127
# im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
# cv2.imwrite('ttd_dewok.jpeg', im_bw)