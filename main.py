from multiprocessing import Process, Queue
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import time

from gui import GUI
from utils import *



def draw(ref_img, image, q):
	# align images
	ref_img = cv2.imread(ref_img)
	warped, h = align_images(image, ref_img)

	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	#ret, thresh1 = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
	warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
	q.put(warped.copy())
	# copy warped image to preserve the original
	warped_copy = warped.copy()

	drawing = False		# control varibale for drawing
	x_prev, y_prev = -1, -1
	start_time = -1
	pts = []			# list of tuples, where each tuple is ([x, y], time)

	def get_warped_coords(x, y):
		# create a zero matrix with the same shape as the original image
		m = np.zeros(image.shape)
		# set the values of the elements in xth row and yth column to 255 in all image channels
		m[y, x, :] = 255
		# get the transformed image matrix
		m = cv2.warpPerspective(m, h, (ref_img.shape[1], ref_img.shape[0]))
		# coordinates of the nonzero elements of m correspond to the coordinates of x, y
		# in the warped (transformed) image
		nonzero = np.nonzero(m)
		if nonzero[0].size == 0 or nonzero[1].size == 0:
			return None
		else:
			y, x = [i[0] for i in nonzero[:-1]]
			return x, y

	# an event listener to draw the laser path
	def draw_line(event, x, y, flags, param):
		nonlocal drawing, x_prev, y_prev, start_time, pts, warped_copy, warped
		if event == cv2.EVENT_LBUTTONDOWN:
			ret = get_warped_coords(x, y)
			if ret is None:
				return

			drawing = True
			warped_copy = warped.copy()
			# set starting_time
			start_time = time.time()
			# get the first point
			x_prev, y_prev = ret
			# add the starting point and its relative time to the list
			pts.append(([x_prev, y_prev], time.time() - start_time))

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				ret = get_warped_coords(x, y)
				if ret is None:
					return
				
				xi, yi = ret
				# draw a line between current and previous points
				cv2.line(warped_copy, (xi, yi), (x_prev, y_prev), (0,255,0), 2)
				#gui.update_image(Image.fromarray(cv2.cvtColor(warped_copy, cv2.COLOR_BGR2RGB)))
				# set previous point to current point
				x_prev, y_prev = xi, yi
				# add current point and its relative time to the list
				pts.append(([x_prev, y_prev], time.time() - start_time))
				q.put((warped_copy.copy(), None))


		elif event == cv2.EVENT_LBUTTONUP and pts:
			drawing = False
			# redraw last 0.5s in red
			end_time = pts[-1][1]
			x_prev, y_prev = pts[-1][0]
			count = 1
			for p in reversed(pts[:-1]):
				if end_time - p[1] >= 0.5:
					break
				
				xi, yi = p[0]
				cv2.line(warped_copy, (xi, yi), (x_prev, y_prev), (0,0,255), 2)
				x_prev, y_prev = xi, yi
				count += 1
			q.put((warped_copy.copy(), pts))
			print("points per sec:", count / 0.5)
			# reset pts
			pts = []

	# display original image and the warped image
	cv2.namedWindow("orig")
	#cv2.namedWindow("warped")
	cv2.setMouseCallback("orig", draw_line)
	cv2.imshow("orig", image)

	while(True):
		#cv2.imshow("warped", warped_copy)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break

	cv2.destroyAllWindows()



def webcam(q, ref_img):
	aiming = False
	stop = False
	prevLoc = None
	frame_draw = None
	pts = []

	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	prev_frame = frame.copy()
	_, h = asift(frame, ref_img)
	# q.put((warped, None))


	while(True):
		# Capture frame-by-frame
		_, frame = cap.read()

		# Display the resulting frame
		cv2.imshow('frame', frame)
		ret, x, y = detect_laser(frame)
		maxLoc = get_warped_coords(x, y, frame, ref_img, h)
		if not aiming and ret:
			aiming = True
			prevLoc = maxLoc
			frame_draw = ref_img.copy()
		
		if aiming:  # i.e laser has started to be seen
			if ret: # if laser is detected
				pts.append((maxLoc, False))
				# if the previous location is None then start drawing from the current maxLoc
				if prevLoc is None:
					prevLoc = maxLoc
				if maxLoc is not None and prevLoc is not None:
					if stop: cv2.line(frame_draw, prevLoc, maxLoc, (0,0,255), 2)
					else: cv2.line(frame_draw, prevLoc, maxLoc, (0,255,0), 2)
				prevLoc = maxLoc
				q.put((frame_draw, None))
			elif not stop:	# if laser is currently not detected
				stop = True
				stop_time = time.time()
				ret2 = False
				while time.time() - stop_time < 1 and not ret2:
					ret, f = cap.read()
					ret2, x2, y2 = detect_laser(f)
				if ret2:
					cv2.circle(frame_draw, prevLoc, 15, (255,0,0), -1)
					q.put((frame_draw, None))
					pts.append((prevLoc, True))
					prevLoc = get_warped_coords(x2, y2, frame, ref_img, h)
				else:
					aiming = False
					stop = False
					pts = []
			else:
				q.put((frame_draw, pts))
				pts = []
				frame_draw = None
				stop = False
				aiming = False

		# break out of the loop if q is pressed		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

root = GUI(img_size=800)
q = Queue()
p = Process(target=webcam, args=(q, cv2.imread(root.target_img)))
p.start()


# def show_selection(event):
# 	selection = root.listbox.curselection()
# 	if selection:
# 		target_copy = root.target.copy()
# 		pts = root.shoots[selection[0]]
# 		prev = pts[0][0]
# 		red = False
# 		for i in range(1, len(pts)):
# 			pt = pts[i]
# 			if pt[1]:
# 				cv2.circle(target_copy, pt[0], 15, (255,0,0), -1)
# 				prev = pts[i+1][0]
# 				red = True
# 			else:
# 				if red: cv2.line(target_copy, prev, pt[0], (0,0,255), 2)
# 				else: cv2.line(target_copy, prev, pt[0], (0,255, 0), 2)
# 				prev = pt[0]

# 		root.update_image(Image.fromarray(cv2.cvtColor(target_copy, cv2.COLOR_BGR2RGB)))


def update_image():
	if not q.empty():
		img, pts = q.get()
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		root.update_image(img)
		if pts is not None:
			now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
			root.shoots[root.count] = pts
			root.count += 1
			root.insert_entry(now)
	root.after(1, update_image)

# root.listbox.bind('<Double-1>', show_selection)
update_image()
root.mainloop()
p.join()


# TO DO: add a control variable and a button for a new shot (?)
# TO DO: 3 secs before shot length
# TO DO: 2  Points
# TO DO: Change the background
# TO DO: Single shot (all on the same screen)
# TO DO: average points (add a button for this)
# TO DO: 1  laser calibration based on the shooting location
# TO DO: 3, 4   find circles and track