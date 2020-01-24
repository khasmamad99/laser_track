import numpy as np
import cv2



def save(name, frame, text):
	cv2.imwrite(name+".jpg", frame)
	open(name+".txt", "w").write(text)



def detect_laser2(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5,5), 0)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)
	print(maxVal)
	if maxVal < 220:
		return None

	offset = 10
	max_x, max_y = maxLoc
	x = max_x - offset
	y = max_y - offset
	h = w = offset * 2

	return (x, y, w, h)


def detect_laser(image, dilate=False, erode=False, subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()):
	# resize image
	h, w, c = image.shape
	scale = min(640/w, 480/h)
	im = cv2.resize(image, (int(w*scale), int(h*scale)))

	# find mask
	mask = subtractor.apply(image, None, learningRate=0)
	kernel = np.ones((5, 5), np.uint8)
	if erode:
		mask = cv2.erode(mask, kernel, iterations=1)
	if dilate:
		mask = cv2.dilate(mask, kernel, iterations=1)

	# cv2.imshow("mask", mask)

	# find avg
	avg = cv2.mean(mask)[0]
	# print(avg)
	while avg > 0.6:
		mask = cv2.erode(mask, kernel, iterations=1)
		avg = cv2.mean(mask)[0]

	else:
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			cnt = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
			cont = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [cnt], 0, (0, 255, 0), 3)
			#cv2.imshow("cont", cont)
			# print("AREA:", cv2.contourArea(cnt))
			if cv2.contourArea(cnt) > 0:
				# M = cv2.moments(cnt)
				# cx = int(M['m10']/M['m00'])
				# cy = int(M['m01']/M['m00'])
				x, y, w, h = cv2.boundingRect(cnt)
				# min_size = 20
				# if w < min_size or h < min_size:
				# 	x_center = x + w/2
				# 	y_center = y + h/2
				# 	x = int(x_center - min_size/2)
				# 	y = int(y_center - min_size/2)
				# 	w = h = min_size
				return (x, y, w, h)

			else:
				return None
		else:
			return None


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0] + box[2]/2.0
    y = box[1] + box[3]/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)



if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	frame_w, frame_h = cap.get(3), cap.get(4)
	frame_id = 0
	divisor = 13
	while(True):
		_, frame = cap.read()
		name = "data/vid_7_" + str(frame_id)
		if frame_id % divisor == 0:
			s = True
		else:
			s = False

		# s = False
		frame_id += 1

		box = detect_laser(frame)
		# box = detect_laser2(frame)
		text = ""
		frame_box = frame.copy()

		if box:
			x, y, w, h = box
			frame_box = cv2.rectangle(frame.copy(),(x,y),(x+w,y+h),(0,255,0),2)
			if s: cv2.imwrite(name+"_box.jpg", frame_box)
			box_norm = convert((frame_w, frame_h), box)
			text = "0 " + " ".join([str(a) for a in box_norm])
		cv2.imshow("frame", frame_box)

		if s:
			# save bgr
			name_bgr = name + "_bgr"
			save(name + "_bgr", frame, text)

			# save rgb
			# save(name+"_rgb", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), text)

			# save gray
			save(name+"_gray", cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), text)

		key = cv2.waitKey(1)
		if key == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()