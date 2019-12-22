import numpy as np
import cv2

cap = cv2.VideoCapture("out_office.avi")

while(True):


    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    detected_circles = cv2.HoughCircles(blur,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 15, minRadius = 3, maxRadius = 10) 

    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 

        # birghtest = 
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2]


    
            # Draw the circumference of the circle. 
            cv2.circle(frame, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(frame, (a, b), 1, (0, 0, 255), 3) 

    #print("finished")        
    cv2.imshow("Detected Circle", frame) 
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()