import cv2


class Controller:
    def __init__(self, target, view_proxy, user):
        self.target = target
        self.view_proxy = view_proxy
        self.user = user


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


    def webcam(self):
        webcam_id = self.set_webcam_id()
        cap = cv2.VideoCapture(webcam_id)
        
        # initialize homogrpahy matrix
        _, frame = cap.read()
        _, h = asift(frame, ref_img)
        assert h is not None, "Could not initialize homography matrix"

        while(True):



    def recalibrate(self):
        pass




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

        cv2.imshow("mask", mask)

        # find avg
        avg = cv2.mean(mask)[0]
        if avg > 1 or avg < 0.0001:
            # if change in the image is too much (change in light/brightness), ignore it
            if avg > 1:
                print("Make sure that the camera is stable!")
            return False, None, None

        else:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]
                cont = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), [cnt], 0, (0, 255, 0), 3)
                cv2.imshow("cont", cont)
                print("AREA:", cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 5:
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    return True, cx, cy

                else:
                    return False, None, None
            else:
                return False, None, None