from multiprocessing import Process, Queue, Value, Manager
import cv2
import time

from oop.Model.LaserDetector import LaserDetector
from oop.Model.Shot import *


class Controller:
    def __init__(self, user, view_contoller):
        self.target = target
        self.user = user
        self.detect_laser = LaserDetector.dl_object_detection
        self.inq = Queue()
        self.outq = Queue()

        self.view_control = Manager.Namespace()
        self.view_control.frame = None
        self.view_control.shot = None
        self.shot_control = Value(int)
        self.shot_control.value = target.shot_control.get()     # WARNING: check naming
        self.calib_control = Value(bool)
        self.calib_control = False

        target_dict = self.view_control.init_target()

        self.target_size = 800      # WARNING: needs to be fixed

        

    def webcam(self):
        webcam_id = self.set_webcam_id()
        self.cap = cv2.VideoCapture(webcam_id)
        
        # initialize homogrpahy matrix
        _, frame = cap.read()
        _, self.h = asift(frame, self.target.img)
        assert h is not None, "Could not initialize homography matrix"

        prev_x, prev_y = None, None
        detected = [False, False, False]
        while(True):
            frame, ret = cap.read()
            # assert ret, "Could not read the frame"
            ret, x, y = self.detect_laser(frame)
            detected = detected.append(ret)[1:]
            if ret:
                q.put((x, y))
                prev_x, prev_y = x, y
            else:
                if True in detected:
                    q.put((prev_x, prev_y))
                else:
                    q.put(None)

    
    def get_trackshot(self):
        aiming = False
        stop = False
        prev_coords = None
        while(True):
            if self.shot_control.value == 1:
                coords = q.get()
                ret = coords is not None
                if ret:
                    # add offset
                    coords = [i - j for i, j in zip(coords, self.target.calibration_offset)]
                    # check if calibrated coords are within image boundaries
                    if coords[0] < or coords[1] < 0 or coords[0] > self.cap.get(3) or coords[1] > self.cap.get(4):
                        coords = None
                    # get warped coords
                    coords = self.real2warped(*coords)

                    if not aiming:
                        aiming = True
                        prev_coords = coords
                        draw_frame = self.target.img.copy()
                        shot = TrackShot(self.target.img_path)
                    
                if aiming:
                    if ret:
                        point = Point(coords, time.time(), False)
                        shot.pts.append(point)
                        if prev_coords is None:
                            prev_coords = coords
                        if coords is not None and prev_coords is not None:
                            # if not shot yet, draw green, otherwise draw red
                            if stop: cv2.line(draw_frame, prevLoc, maxLoc, (0,0,255), 2)
                            else: cv2.line(draw_frame, prevLoc, maxLoc, (0,255,0), 2)
                        prev_coords = coords
                        outq.put((draw_frame, None))
                    elif not stop:
                        stop = True
                        stop_time = time.time()
                        ret2 = False
                        while time.time() - stop_time < 1 and not ret2:
                            if not q.empty():
                                coords2 = q.get()  
                                ret2 = coords2 is not None
                        if ret2 and prev_coords is not None:
                            point = Point(coords, time.time(), True)
                            cv2.circle(draw_frame, prev_coords, 15, (255, 0, 0), -1)
                            shot.score = self.target.calc_score(self.target, *coords)
                            shot.distance = self.target.calc_distance(shot.pts)
                            self.draw_stats(fr, shot)
                            outq.put((draw_frame, None))
                        else:
                            aiming = False
                            stop = False
                    else:
                        outq.put((draw_frame, shot))
                        aiming = False
                        stop = False


    def get_singleshot(self):
        draw_frame = self.target.img.copy()
        while(True):
            if self.shot_control.value == 2:
                coords = q.get()
                if coords:
                    # add offset
                    coords = [i - j for i, j in zip(coords, self.target.calibration_offset)]
                    # check if calibrated coords are within image boundaries
                    if coords[0] < or coords[1] < 0 or coords[0] > self.cap.get(3) or coords[1] > self.cap.get(4): continue
                    # get warped coords
                    coords = self.real2warped(*coords)
                    # check if warped coords are within the boundaries of the warped image
                    if coords is None: continue
                    score = self.target.calc_score(self.target, *coords)
                    shot = SingelShot(score, self.target.img_path)
                    cv2.circle(draw_frame, coords, 15, (255,0,0), -1)
                    self.draw_stats(draw_frame, shot)
                    self.outq.put((draw_frame, shot))
            elif draw_frame != self.target_img:
                draw_frame = self.target_img.copy()
    

    def update_attrs(self):
        while(True):
            if not self.outq.empty():
                frame, shot = self.outq.get()    # WARNING: check naming
                self.view_attributes.frame = frame
                if self.calib_control.value:
                    self.shot_control.value = 0
                    while not self.outq.empty():
                        outq.get()
                    self.view_attributes.shot = None
                    self.set_calibration_offset(*shot.coords)
                    # display a pop up which destroys itself after 3 secs
                    # self.shot_control.value = self.view. rb value
                else:
                    self.view_attributes.shot = shot

                    

    def view_selection(self, event):
        self.shot_control.value = 0
        # TO DO: enable the "new shot" button
        # clear the out queue
        while not self.outq.empty():
            outq.get()
        self.display_selection()


    def get_new_shot(self, event):
        # disable "new shot" button
        # clear display
        self.outq.put((self.target.img, None))
        # self.shot_control.value = self.view.radiobutton value



    def display_selection(self):
        selection = self.view.listbox.curselection()
        if selection:
            shot = cv2.imread(self.user._shots[selection[0]])
            target = cv2.imread(shot.target_img)
            pts = shot.pts
            prev = pts[0].coords
            red = False
            for i in range(len(pts)):
                pt = pts[i]
                if prev is None:
                    prev = pt.coords
                if pt.coords is not None and prev is not None:
                    if pt.is_shot:
                        cv2.circle(target, pt[0], 15, (255,0,0), -1)
                        draw_stats(target, shot)
                        prev = None
                        red = True
                    else:
                        if red: cv2.line(target, prev, pt.coords, (0,0,255),2)
                        else: cv2.line(target, prev, pt.coords, (0,255,0), 2)
                        prev = pt.coords

            self.outq.put((target, None))

    
    def recalibrate(self, event):
        self.shot_control = 0
        # clear the queue
        self.outq.put((self.target.img, None))
        self.target.calibration_offset = [0,0]
        # display a pop up
        self.calib_control = True
        # self.shot_control = self.view. radiobutton values


    def set_webcam_id(self):
        i = 5
        while i >= -1:
            print(i)
            try:
                cv2.VideoCapture(i)
                break
            except:
                i -= 1
        assert i >= 0, "Could not connect to webcam"
        return i

    
    def real2warped(self, x, y):
        # create a zero matrix with the same shape as the original image
        m = np.zeros((self.cap.get(4), self.cap.get(3), 3))
        # set the values of the elements in xth row and yth column to 255 in all image channels
        m[y, x, :] = 255
        # get the transformed image matrix
        m = cv2.warpPerspective(m, self.h, (self.target.img.shape[1], self.target.img.shape[0]))
        # coordinates of the nonzero elements of m correspond to the coordinates of x, y
        # in the warped (transformed) image
        nonzero = np.nonzero(m)
        if nonzero[0].size == 0 or nonzero[1].size == 0:
            return None
        else:
            y, x = [i[0] for i in nonzero[:-1]]
            return x, y

    
    def warped2real(self, x, y):
        m = np.zeros(self.target.img.shape)
        m[y, x, :] = 255
        m = cv2.warpPerspective(m, np.linalg.inv(self.h), (self.cap.get(3), self.cap.get(4)))
        nonzero = np.nonzero(m)
        if nonzero[0].size == 0 or nonzero[1].size == 0:
            return None
        else:
            y, x = [i[0] for i in nonzero[:-1]]
            return x, y

    
    def set_calibration_offset(self, x, y):
        real_center_coords = self.warped2real(*self.target.center_coords)
        self.target.calibration_offset = [i - j for i, j in zip([x, y], real_center_coords)]


    def draw_shot_stats(self, frame, shot):)
        org_y = frame.shape[0] - 10
        org_x = 10
        cv2.putText(frame, str(shot), (int(org_x), int(org_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


