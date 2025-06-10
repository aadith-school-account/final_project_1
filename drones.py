#imports
import argparse
import imutils
import cv2

#when video updated, use
#python detect_targets.py --video my_video.mp4



#argument + parsing, takes input from CLI instead of hardcoding
#MORE flexible than hardcoding path, input through bash when program starts
ap = argparse.ArgumentParser() #create a parser to allow input via CLI
ap.add_argument("--v", "--video", help="path to the video file") #allows user to specify video path
args = vars(ap.parse_args()) #converts parsed arguments into dictionary

#load vid through cv2 using argparse
camera = cv2.VideoCapture(args["video"])

#constant loop
while True:
    #grab current frame, init status text
    (grabbed, frame) = camera.read() #returns tuple of 2 values
    #grabbed is bool val
    #frame is NumPY array of size NxM pixels, which is processed and we try to find targets
    status = "No Targets" #set a string to a var

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



    #loop over the contoured objects, cnts is list
    for c in cnts:
        #aprx the contour
        peri = cv2.arcLength(c, True) #calc perimiter of contour
        approx = cv2.approxPolyDB (c, 0.01 * peri, True) #simplifiy contour into fewer points

        #ensure that aprx contour is "roughly" rectangular
        if len(approx) >= 4 and len(approx) <= 6: #if it doesn't fit, this is ignored
            #compute bounding box of the approximated contoure
            #use the bounding box to compute aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx) #finds rectangle to surround shape
            aspectRatio = w / float(h)

            #compute solidiity of og contour
            area = cv2.contourArea(c)
            hullArea = cv2.contourArea(cv2.convexHull(c))
            solidity = area / float(hullArea)

            #compute wether or not width, height, solidity and aspect ratio
            #fall within bounds
            keepDims = w > 25 and h > 25
            keepSolidity = solidity > 0.9
            keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

            #ensure contour passess all of ^ tests
            if keepDims and keepSolidity and keepAspectRatio:
                #draw outline around target, update status
                #text
                cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
                status = "Target(s) Acquired"

                #compute center of contour region, draw crossharis
                M = cv2.moments(approx) #find center, cX and cY are center coordinates

                #this massive section draws the crossharis at the center. Ripped off of article.
                (cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
                (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
                (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
                cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
                cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)

    #draw the status text on frame
    cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)

    #show frame, record if key pressed
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #if 'q' is pressed, stop loop
    if key == ord("q"):
        break


#cleanup
camera.release()
cv2.destroyAllWindows()


