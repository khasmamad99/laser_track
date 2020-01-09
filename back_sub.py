import numpy as np 
import cv2 
  
cap = cv2.VideoCapture(2) 
cap.set(3, 640)
cap.set(4, 480)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 

def letterbox(image, size):
    ih, iw, ic = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw, nh))
    return image


lower = [0,50, 0]
upper = [60, 255, 60]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")



sm = 0
count = 0 
mx = 0
mn = 999999999
while(1): 
    ret, frame = cap.read()
    mask = cv2.inRange(frame, lower, upper)
    ouput = cv2.bitwise_and(frame, frame, mask=mask)

    h, w, c = frame.shape
    frame = letterbox(frame, (640, 480))
    fgmask = fgbg.apply(frame)
    avg = cv2.mean(fgmask)[0]
    avg_thershold = 1
    #print("AVG:",  avg)
    # print(fgmask)
    if avg < avg_thershold:
        #erode = cv2.erode(fgmask, np.ones((2,2), np.uint8), iterations=1) 
        #dilate = cv2.dilate(erode, np.ones((3,3), np.uint8), iterations=1)

        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse=True)
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            sm += area
            if area > mx:
                mx = area
            if area < mn:
                mn = area
            count += 1
            if area > 10:
                cont = cv2.drawContours(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR), [cnt], 0, (0,255,0), 3)
                cv2.imshow("cont", cont)
            print(area)
















    #contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    #cont = cv2.drawContours(fgmask.copy(), contours, -1, (0,255,0), 3)
    # detected_circles = cv2.HoughCircles(fgmask,  
    #             cv2.HOUGH_GRADIENT, 1, int(min(w, h)*0.9), param1 = 50, 
    #             param2 = 5, minRadius = 2, maxRadius = 20)

    #print(detected_circles)

    # if detected_circles is not None:
    #     detected_circles = np.uint16(np.around(detected_circles))
    #     for pt in detected_circles[0, :]: 
    #         a, b, r = pt[0], pt[1], pt[2]


    
            # Draw the circumference of the circle. 
            #cv2.circle(frame, (a, b), 5, (0, 255, 0), -1) 
    
            # Draw a small circle (of radius 1) to show the center. 
            #cv2.circle(frame, (a, b), 1, (0, 0, 255), 3) 
        # print((detected_circles))
   
    cv2.imshow('frame', frame) 
    cv2.imshow('fgmask', fgmask)
    #cv2.imshow('dil1', erode)
    #cv2.imshow("color", mask)
    #cv2.imshow('cont', cont)
  
      
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
      
print("avg:", sm/count, "min:", mn, "max:", mx)
cap.release() 
cv2.destroyAllWindows() 