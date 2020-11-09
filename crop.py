import numpy as np
import cv2
import matplotlib.pyplot as plt

fPath = "/home/ajey/Desktop/t_Page_0035.jpg"
img = cv2.imread(fPath)
th, imag = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
edges = cv2.Canny(imag,100,200)
dkernel = np.ones((35,15), np.uint8)
dilation = cv2.dilate(edges,dkernel,iterations = 1)
ckernel = np.ones((20,15), np.uint8)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, ckernel)
image, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(contours):
	x,y,w,h = cv2.boundingRect(cnt)
	if w*h < 40000:
		continue
	else:
		img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,198),10)
plt.figure(figsize=(15,15))
plt.imshow(img)
plt.yticks([])
plt.xticks([])
plt.show()