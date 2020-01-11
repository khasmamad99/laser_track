from multiprocessing import Process, Queue, Value
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import time

from gui import GUI, DialogRecalib
from utils import *



def webcam(q, rb_value, recalibrate, img_size, target):
	aiming = False
	stop = False
	prevLoc = None
	frame_draw = None
	pts = []
	offset = [0, 0]
	prev_rb_value = rb_value.value
	prev_recalibrate = recalibrate.value
	ref_img = cv2.imread(target["img_path"])
	target_conts = np.load(target["contours_npy"], allow_pickle=True)
	target_center = target["center_coords"][0] / ref_img.shape[1] * root.img_size, target["center_coords"][1] / ref_img.shape[0] * root.img_size

	cap = cv2.VideoCapture(0)
	_, frame = cap.read()
	_, h = asift(frame, ref_img)
	assert h is not None, "Could not initialize the homography matrix"


	while(True):
		# Capture frame-by-frame
		_, frame = cap.read()

		# Display the resulting frame
		cv2.imshow('frame', frame)
		ret, x, y = detect_laser(frame, dilate=True)

		# reset offset to 0,0 before recalibration
		if recalibrate.value == 1: # synch of processes problem?
			offset = [0,0]

		# rb_value (RadioButton value)
		# 1 : tracking, 0 : one-shot
		# if there is a change in rb_value or recalibration is started/finished, then reset the background image
		if rb_value.value != prev_rb_value or (recalibrate.value != prev_recalibrate and recalibrate.value != 2):
			frame_draw = ref_img.copy()
			aiming = False
			stop = False
			prev_rb_value = rb_value.value
			prev_recalibrate = recalibrate.value
			q.put((frame_draw, None))
		
		# tracking
		if rb_value.value == 1:
			# transform the coordinates into warped image plane
			# maxLoc can be None, if x and y are out of the scope of the warped image plane
			maxLoc = get_warped_coords(x, y, frame, ref_img, h)
			# add calibration offset
			maxLoc = tuple([i - j for i, j in zip(list(maxLoc), list(offset))]) if maxLoc is not None else None
			# if laser is seen for the first time
			if not aiming and ret:
				aiming = True
				prevLoc = maxLoc
				frame_draw = ref_img.copy()
			
			# laser has started to be seen (it might or might not be visible currently, the important thing is it was visible very recently)
			if aiming:
				# laser is currently visible/is detected
				if ret:
					pts.append((maxLoc, False, None, time.time()))
					# if the previous location is None then start drawing from the current maxLoc
					# laser can go out of the scope of the warped image plane
					# in this case drawing is started from where the laser enters the warped image plane
					if prevLoc is None:
						prevLoc = maxLoc
					if maxLoc is not None and prevLoc is not None:
						# if not shot yet, draw green, otherwise draw red
						if stop: cv2.line(frame_draw, prevLoc, maxLoc, (0,0,255), 2)
						else: cv2.line(frame_draw, prevLoc, maxLoc, (0,255,0), 2)
					prevLoc = maxLoc
					q.put((frame_draw, None))
				# if laser is currently not visible/not detected and it's the first time the laser has become invisible
				elif not stop:
					stop = True
					stop_time = time.time()
					ret2 = False
					# check if laser gets detected within the next second
					while time.time() - stop_time < 1 and not ret2:
						_, f = cap.read()
						ret2, x2, y2 = detect_laser(f)
					# if laser gets detected within the given time and prevLoc is not None
					if ret2 and prevLoc is not None:
						scr = None
						# draw the circle in the location, where the laser became invisible
						cv2.circle(frame_draw, prevLoc, 15, (255,0,0), -1)
						# set the offset if recalibration button is pressed/recalibration value is 1
						if recalibrate.value == 1:
							offset = [i - j for i, j in zip(list(prevLoc), target["center_coords"])] # TO DO: change this to a variable
							recalibrate.value = 2
						# if not recalibrating, draw the score and distance
						else:
							scr = calc_score(target_conts, target["name"], *prevLoc)
							pts.append((prevLoc, True, scr, time.time()))
							distance = calc_distance(pts, target["real_size"], (img_size, img_size))
							draw_score(frame_draw, scr, distance)
						q.put((frame_draw, None))
						prevLoc = get_warped_coords(x2, y2, frame, ref_img, h)
					else:
						aiming = False
						stop = False
						pts = []
				else:
					if recalibrate.value == 0:
						q.put((frame_draw, pts))
					else:
						q.put((frame_draw, None))
					pts = []
					frame_draw = None
					stop = False
					aiming = False
		elif rb_value.value == 0:
			if not aiming and ret:
				aiming = True
				maxLoc = get_warped_coords(x, y, frame, ref_img, h)
				if maxLoc is not None:
					maxLoc = tuple([i - j for i, j in zip(list(maxLoc), list(offset))])
					cv2.circle(frame_draw, maxLoc, 15, (255,0,0), -1)
					# do not save image (input None instead of pts) if recalibrated
					if recalibrate.value == 0:
						scr = calc_score(target_conts, target["name"], *maxLoc)
						frame_draw_cpy = frame_draw.copy()
						draw_score(frame_draw_cpy, scr, None)
						q.put((frame_draw_cpy, [(maxLoc, True, scr, None)]))
					elif recalibrate.value == 1:
						# recalibrate
						q.put((frame_draw, None))
						offset = [i - j for i, j in zip(list(maxLoc), target["center_coords"])]
						recalibrate.value = 2

			if not ret:
				aiming = False
				
		# break out of the loop if q is pressed		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


def update_gui():
	while not q.empty():
		img, pts = q.get()
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		root.update_image(img)
		if pts is not None:
			now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
			root.insert_entry(now, pts)

	if recalibrate.value == 2:
		img, pts = q.get()
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		root.update_image(img)
		dial = DialogRecalib(root, title="Recalibration", text="Recalibration is done!")
		root.wait_window(dial.top)
		root.recalibrate = 0
		recalibrate.value = 0
	rb_value.value = root.rb_value.get()
	recalibrate.value = root.recalibrate # 0 : not recalibrating; 1 : recalibrating; 2 : recalibration done

	root.after(1, update_gui)



if __name__ == "__main__":
	root = GUI(img_size=800)
	q = Queue()
	recalibrate = Value('i', root.recalibrate)
	rb_value = Value('i', root.rb_value.get())
	p = Process(target=webcam, args=(q, rb_value, recalibrate, root.img_size, root.target))
	p.start()
	update_gui()
	root.mainloop()
	p.join()


# TO DO: add a control variable and a button for a new shot (?)
# TO DO: 3 secs before shot length
# TO DO: 2 Points
# TO DO: Change the background
# DONE : Single shot (all on the same screen)
# TO DO: average points (add a button for this)
# DONE : 1  laser calibration based on the shooting location
# TO DO: 3, 4   find circles and track
# TO DO: add offset to the real coords?