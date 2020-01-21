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
        while(True):
            if self.shot_control.value == 1:
                # do trackshot
                pass

    def get_singleshot(self):
        while(True):
            if self.shot_control.value == 2:
                coords = q.get()
                if coords:
                    draw_frame = self.target.img.copy()
                    # preprocess
                    coords = self.get_warped_coords(*coords)
                    # add offset
                    coords = [i - j for i, j in zip(coords, self.target.calibration_offset)]
                    score = self.target.calc_score(self.target, *coords)
                    shot = SingelShot(score, self.target.img_path)
                    cv2.circle(draw_frame, coords, 15, (255,0,0), -1)
                    self.outq.put((draw_frame, shot))

    
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

    
    def get_warped_coords(self, x, y):
        # create a zero matrix with the same shape as the original image
        m = np.zeros(self.cap.get(4), self.cap.get(3), 3)
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

