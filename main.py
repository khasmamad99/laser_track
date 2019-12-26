from multiprocessing import Process, Queue
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import time

from gui import GUI
from utils import *



def webcam(q, ref_img):
	aiming = False
	stop = False
	prevLoc = None
	frame_draw = None
	pts = []

	cap = cv2.VideoCapture(2)
	ret, frame = cap.read()
	_, h = asift(frame, ref_img)

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
				while time.time() - stop_time < 1 and not ret2: # check if laser gets detected within the next second
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



if __name__ == "__main__":
	root = GUI(img_size=800)
	q = Queue()
	p = Process(target=webcam, args=(q, cv2.imread(root.target_img)))
	p.start()
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