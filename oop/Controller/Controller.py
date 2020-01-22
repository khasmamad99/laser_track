from multiprocessing import Process, Queue, Value, Manager
import cv2

from oop.Model.LaserDetector import LaserDetector
from oop.Model.Shot import *


class Controller:
    def __init__(self, target, user):
        self.target = target
        self.user = user
        self.detect_laser = LaserDetector.dl_object_detection
        self.inq = Queue()
        self.outq = Queue()

        self.view_attrs = Manager.Namespace()
        self.view_attrs.frame = None
        self.view_attrs.shot = None
        self.shot_control = Value(int)
        self.shot_control.value = target.shot_control.get()     # WARNING: check naming
        self.calib_control = Value(bool)
        self.calib_control = False
        self.target_size = 600      # WARNING: needs to be fixed

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

                    if not aiming and ret:
                        aiming = True
                        prev_coords = coords
                        frame_draw = self.target.img.copy()



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
                    self.outq.put((draw_frame, shot))
            elif draw_frame != self.target_img:
                draw_frame = self.target_img.copy()
    

    def update_attrs(self):
        while(True):
            if not self.outq.empty():
                frame, shot = self.outq.get()    # WARNING: check naming
                self.view_attributes.frame = frame
                self.view_attributes.shot = None
                if self.calib_control.value:
                    self.view_attributes.shot = shot
                    

    
    def set_calib_control(self, val):
        if self.calib_control.val != val:
            self.view.update_frame(self.target.img) 


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
        m = np.zeros(self.cap.get(4), self.cap.get(3), 3)
        # set the values of the elements in xth row and yth column to 255 in all image channels
        m[y, x, :] = 255
        # get the transformed image matrix
        m = cv2.warpPerspective(m, self.h, (self.target.target_size, self.target_size)
        # coordinates of the nonzero elements of m correspond to the coordinates of x, y
        # in the warped (transformed) image
        nonzero = np.nonzero(m)
        if nonzero[0].size == 0 or nonzero[1].size == 0:
            return None
        else:
            y, x = [i[0] for i in nonzero[:-1]]
            return x, y

    
    def warped2real(self, x, y):
        m = np.zeros(self.target_size, self.target_size, 3)
        m[y, x, :] = 255
        m = cv2.warpPerspective(m, np.linalg.inv(self.h), (self.cap.get(3), self.cap.get(4)))
        nonzero = np.nonzero(m)
        if nonzero[0].size == 0 or nonzero[1].size == 0:
            return None
        else:
            y, x = [i[0] for i in nonzero[:-1]]
            return x, y

    
    def recalibrate(self, x, y):
        real_center_coords = warped2real(*self.target.center_coords)
        self.target.calibration_offset = [i - j for i, j in zip([x, y], real_center_coords)]



