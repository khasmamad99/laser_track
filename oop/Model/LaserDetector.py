import cv2
import numpy as np

class LaserDetector:

    def bg_subtraction(self, image, dilate=False, erode=False, subtractor=cv2.bgsegm.createBackgroundSubtractorMOG()):
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

    
    def dl_object_detection(self, frame, net = cv2.dnn.readNet("tracker/yolov3-tiny-1cls_best.weights", "tracker/yolov3-tiny-1cls.cfg"), classes = ["laser"]):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        height, width, channels = frame.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.01:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3, 1)
                    mx = -1
                    mx_i = -1
                    for i in indexes:
                        i = i[0]
                        if confidences[i] > mx:
                            mx = confidences[i]
                            mx_i = i
                    x, y, w, h = boxes[mx_i]
                    center_x = int(x+w/2)
                    center_y = int(y+h/2)
                    return True, center_x, center_y

        return False, None, None       