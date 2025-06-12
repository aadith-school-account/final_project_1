#imports
import argparse
import imutils
import cv2

#website
#https://pyimagesearch.com/2015/05/04/target-acquired-finding-targets-in-drone-and-quadcopter-video-streams-using-python-and-opencv/


#when video updated, use
#python detect_targets.py --video my_video.mp4

def nothing(x):
    pass

cv2.namedWindow("Controls") 
cv2.createTrackbar("Min Area", "Controls", 500, 10000, nothing)   # min area to detect shapes
cv2.createTrackbar("Canny Lower", "Controls", 50, 500, nothing)   # lower Canny threshold
cv2.createTrackbar("Canny Upper", "Controls", 150, 500, nothing)  # upper Canny threshold


#argument + parsing, takes input from CLI instead of hardcoding
#MORE flexible than hardcoding path, input through bash when program starts
ap = argparse.ArgumentParser() #create a parser to allow input via CLI
ap.add_argument("--video", help="path to the video file") #allows user to specify video path
args = vars(ap.parse_args()) #converts parsed arguments into dictionary

#load vid through cv2 using argparse
#camera = cv2.VideoCapture(0 if args["video"] is None else args["video"])
camera = cv2.VideoCapture(0)

#constant loop
while True:
    #grab current frame, init status text
    (grabbed, frame) = camera.read() #returns tuple of 2 values
    #grabbed is bool val
    #frame is NumPY array of size NxM pixels, which is processed and we try to find targets
    status = "No Targets" #set a string to a var

    # Get values from trackbars
    min_area = cv2.getTrackbarPos("Min Area", "Controls")
    canny_lower = cv2.getTrackbarPos("Canny Lower", "Controls")
    canny_upper = cv2.getTrackbarPos("Canny Upper", "Controls")

    # Edge detection with dynamic thresholds
    #check to see if reached end of vid
    if not grabbed:
        break

    #weird part, stolen from article (convert frame to grayscale, blur and detect edges)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blurred =cv2.GaussianBlur(gray, (7, 7), 0) #apply gausian blur
    edged = cv2.Canny(blurred, 50, 150) #edge detection

    #find contours? Outline of objects detected in images
    #finds shapes in a frame, and simplify them
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)



    edged = cv2.Canny(blurred, canny_lower, canny_upper)

    #loop over the contoured objects, cnts is list
    for c in cnts:
        #aprx the contour
        if cv2.contourArea(c) < 500:
            continue
        if cv2.contourArea(c) < min_area:
            continue

        peri = cv2.arcLength(c, True) #calc perimiter of contour
        approx = cv2.approxPolyDP (c, 0.02 * peri, True) #simplifiy contour into fewer points


        shape = "Unidentified"
        #ensure that aprx contour is "roughly" rectangular

        if len(approx) == 3:
            shape = "Triangle"

        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"

        elif len(approx) == 5:
            shape = "Pentagon"

        elif len(approx) > 5:
            shape = "Circle"
            area = cv2.contourArea(c)
            if peri == 0:
                continue  # avoid division by zero
            circularity = 4 * 3.1416 * (area / (peri * peri))

            if circularity > 0.7:
                shape = "Circle"
            else:
                shape = "Polygon"

        # Draw the contour and shape label
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # Get the center of the shape to place the text
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        cv2.putText(frame, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(frame, f"Min Area: {min_area}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    status = "Target(s) Acquired"
    #show frame, record if key pressed
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #if 'q' is pressed, stop loop
    if key == ord("q"):
        break







#cleanup
camera.release()
cv2.destroyAllWindows()


