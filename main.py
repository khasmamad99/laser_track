from multiprocessing import Process, Queue, Value
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import time

from gui import GUI, DialogRecalib
from utils import *



def webcam(q, rb_value, recalibrate, ref_img):
	aiming = False
	stop = False
	prevLoc = None
	frame_draw = None
	pts = []
	offset = [0, 0]
	prev_rb_value = rb_value.value
	prev_recalibrate = recalibrate.value

	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	_, h = asift(frame, ref_img)
	# lb = np.array(letterbox_image(Image.fromarray(ref_img), (800, 800)))
	# cv2.imshow("lb", lb)


	while(True):
		# Capture frame-by-frame
		_, frame = cap.read()

		# Display the resulting frame
		cv2.imshow('frame', frame)
		ret, x, y = detect_laser(frame)

		print(recalibrate.value)
		if recalibrate.value == 1: # synch of prcesses problem?
			offset = [0,0]

		if rb_value.value != prev_rb_value or (recalibrate.value != prev_recalibrate and recalibrate.value != 2):
			frame_draw = ref_img.copy()
			aiming = False
			stop = False
			prev_rb_value = rb_value.value
			prev_recalibrate = recalibrate.value
			q.put((frame_draw, None))
		
		print("PREV RECALIB:", prev_recalibrate)
		
		if rb_value.value == 1: # track
			maxLoc = get_warped_coords(x, y, frame, ref_img, h)
			# add calibration offset
			maxLoc = tuple([i - j for i, j in zip(list(maxLoc), list(offset))]) if maxLoc is not None else None
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
					while time.time() - stop_time < 1 and not ret2: # check if laser gets detected within the next second
						ret, f = cap.read()
						ret2, x2, y2 = detect_laser(f)
					if ret2:
						if recalibrate.value == 1:
							offset = [i - j for i, j in zip(list(prevLoc), [400, 400])] # TO DO: change this to a variable
							recalibrate.value = 2
						cv2.circle(frame_draw, prevLoc, 15, (255,0,0), -1)
						q.put((frame_draw, None))
						pts.append((prevLoc, True))
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
						q.put((frame_draw, [((x, y), True)]))
					elif recalibrate.value == 1:
						# recalibrate
						q.put((frame_draw, None))
						offset = [i - j for i, j in zip(list(maxLoc), [400, 400])] # TO DO: change this to a variable
						recalibrate.value = 2

			if not ret:
				aiming = False
				
		# break out of the loop if q is pressed		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


def update_image():
	while not q.empty():
		img, pts = q.get()
		img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		root.update_image(img)
		if pts is not None:
			now = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
			root.shoots[root.count] = pts
			root.count += 1
			root.insert_entry(now)

	if recalibrate.value == 2:
		dial = DialogRecalib(root, title="Recalibration", text="Recalibration is done!")
		root.wait_window(dial.top)
		root.recalibrate = 0
		recalibrate.value = 0
	rb_value.value = root.rb_value.get()
	recalibrate.value = root.recalibrate # 0 : not recalibrating; 1 : recalibrating; 2 : recalibration done

	root.after(1, update_image)



if __name__ == "__main__":
	root = GUI(img_size=800)
	q = Queue()
	recalibrate = Value('i', root.recalibrate)
	rb_value = Value('i', root.rb_value.get())
	p = Process(target=webcam, args=(q, rb_value, recalibrate, cv2.imread(root.target_img)))
	p.start()
	update_image()
	root.mainloop()
	p.join()


# TO DO: add a control variable and a button for a new shot (?)
# TO DO: 3 secs before shot length
# TO DO: 2  Points
# TO DO: Change the background
# DONE : Single shot (all on the same screen)
# TO DO: average points (add a button for this)
# TO DO: 1  laser calibration based on the shooting location
# TO DO: 3, 4   find circles and track