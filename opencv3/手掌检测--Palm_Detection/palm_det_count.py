import cv2
import numpy as np
def palm_detection(img_path):
	image = cv2.imread(img_path)
	# 1.方式1：轉到hsv空間獲取手掌mask-多一个缺陷点
	# hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# lower = np.array([0, 20, 80], dtype = 'uint8')  # 色度，亮度，饱和度
	# upper = np.array([45, 255, 255], dtype = 'uint8')
	# skinRegionHSV = cv2.inRange(hsvim, lower, upper)
	# blurred = cv2.blur(skinRegionHSV, (2, 2))  # 均值滤波
	# ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)  # 二值化
	# 1.方式2：直接转灰度图gray-well
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 二值化(取反)
	cv2.imwrite('binary.png', thresh)
	# 2.获取轮廓
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = max(contours, key = lambda x: cv2.contourArea(x))
	cv2.drawContours(image, [contours], -1, (255, 255, 0), 2)
	# 3.凸包检测
	hull = cv2.convexHull(contours)
	cv2.drawContours(image, [hull], -1, (0, 255, 255), 2)
	# 4.凸缺陷检测
	hull = cv2.convexHull(contours, returnPoints=False)
	defects = cv2.convexityDefects(contours, hull)
	print(defects)
	# 画出所有的缺陷点	
	#for i in range(defects.shape[0]):
	#	s, e, f, d = defects[i, 0]
	#	start = tuple(contours[s][0])
	#	end = tuple(contours[e][0])
	#	far = tuple(contours[f][0])
	#	cv2.line(image, start, end, [0, 255, 0], 2)
	#	cv2.circle(image, far, 5, [0, 0, 255], -1)	
	# 5.判断有效缺陷点
	cnt = 0
	if defects is not None:
		for i in range(defects.shape[0]):
			s, e, f, d = defects[i, 0]
			start = tuple(contours[s][0])
			end = tuple(contours[e][0])
			far = tuple(contours[f][0])
			# 余弦定理
			a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
			b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
			c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
			angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # cosine theorem
			if angle <= np.pi / 2: # angle less than 90 degree, treat as fingers 角度大小判断为手指间的缺陷点
				cnt += 1
				cv2.circle(image, far, 4, [0, 0, 255], -1)		
			cv2.putText(image, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)		
	print(cnt)  # 缺陷点个数
	return image
    
img_path = 'palm1.jpg'
det_result = palm_detection(img_path)
cv2.imwrite('palm1_defect_result.png', det_result)
