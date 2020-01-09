from multiprocessing import Process, Queue, Value
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import time

from gui import GUI, DialogRecalib
from utils import *


def webcam(q, id = 0, show = False):
	cap = cv2.VideoCapture(id)
	while(True):
		r, frame = cap.read()
		if not r:
			break

		if show:
			cv2.imshow('frame', frame)
			cv2.waitKey(1)

		ret, x, y = detect_laser(frame)
		q.put((frame, ret, x, y))

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()




def draw(root, q):
	aiming = False
	stop = False
	prevLoc = None
	frame_draw = None
	pts = []
	offset = [0, 0]
	prev_rb_value = root.rb_value.get()
	prev_recalibrate = root.recalibrate
	target_center = [int(root.img_size/2), int(root.img_size/2)]
	target_conts = root.target_conts
	ref_img = cv2.imread(root.target_img)
	h = None

	if not q.empty():
		frame, _, _, _ = q.get()
		_, h = asift(frame, ref_img)

	assert h is not None, "Could not initialize the homography matrix"

	def update_gui():
		nonlocal prev_rb_value, prev_recalibrate, aiming, stop, prevLoc, frame_draw, pts, offset, target_center, target_conts, ref_img, h
		while not q.empty():
			frame, ret, x, y = q.get()
			# reset offset to 0,0 before recalibration
			if root.root.recalibrate == 1:
				offset = [0,0]

			# rb_value (RadioButton value)
			# 1 : tracking, 0 : one-shot
			# if there is a change in rb_value or recalibration is started/finished, then reset the background image
			if root.rb_value.get() != prev_rb_value or (root.recalibrate != root.recalibrate and root.recalibrate != 2):
				frame_draw = ref_img.copy()
				aiming = False
				stop = False
				prev_rb_value = root.rb_value.get()
				prev_root.recalibrate = root.recalibrate
				root.update_image(Image.fromarray(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)))
			
			# tracking
			if root.rb_value.get() == 1:
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
							if stop: cv2.line(frame_draw, prevLoc, maxLoc, (0,0,255), 2)
							else: cv2.line(frame_draw, prevLoc, maxLoc, (0,255,0), 2)
						prevLoc = maxLoc
						root.update_image(Image.fromarray(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)))
					# if laser is currently not visible/not detected and it's the first time the laser has become invisible
					elif not stop:
						stop = True
						stop_time = time.time()
						ret2 = False
						# check if laser gets detected within the next second
						while time.time() - stop_time < 1 and not ret2:
							_, ret2, x2, y2 = q.get()
						# if laser gets detected within the given time and prevLoc is not None
						if ret2 and prevLoc is not None:
							scr = None
							# draw the circle in the location, where the laser became invisible
							cv2.circle(frame_draw, prevLoc, 15, (255,0,0), -1)
							# set the offset if recalibration button is pressed/recalibration value is 1
							if root.recalibrate == 1:
								offset = [i - j for i, j in zip(list(prevLoc), list(target_center))] # TO DO: change this to a variable
								root.update_image(Image.fromarray(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)))
								dial = DialogRecalib(root, title="Recalibration", text="Recalibration is done!")
								root.wait_window(dial.top)
								root.recalibrate = 0
							# if not recalibrating, draw the score and distance
							else:
								scr = calc_score(target_conts, *prevLoc)
								pts.append((prevLoc, True, scr, time.time()))
								distance = calc_distance(pts)
								draw_score(frame_draw, scr, distance)
							root.update_image(Image.fromarray(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)))
							prevLoc = get_warped_coords(x2, y2, frame, ref_img, h)
						else:
							aiming = False
							stop = False
							pts = []
					else:
						if root.recalibrate == 0:
							now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
							root.insert_entry(now, pts)
						root.update_image(Image.fromarray(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)))
						pts = []
						frame_draw = None
						stop = False
						aiming = False
			elif root.rb_value.get() == 0:
				if not aiming and ret:
					aiming = True
					maxLoc = get_warped_coords(x, y, frame, ref_img, h)
					if maxLoc is not None:
						maxLoc = tuple([i - j for i, j in zip(list(maxLoc), list(offset))])
						cv2.circle(frame_draw, maxLoc, 15, (255,0,0), -1)
						# do not save image (input None instead of pts) if root.recalibrated
						if root.recalibrate == 0:
							scr = calc_score(target_conts, *maxLoc)
							frame_draw_cpy = frame_draw.copy()
							draw_score(frame_draw_cpy, scr, None)
							pts = [(maxLoc, True, scr, None)]
							now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
							root.insert_entry(now, pts)
							
						elif root.recalibrate == 1:
							# root.recalibrate
							offset = [i - j for i, j in zip(list(maxLoc), list(target_center))]
							root.recalibrate = 2

						root.update_image(Image.fromarray(cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)))

				if not ret:
					aiming = False
				