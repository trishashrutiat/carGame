import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from directkeys import PressKey, A, D, Space, ReleaseKey

try:
    # Start the video stream
    cam = VideoStream(src=0).start()
    currentKey = list()

    while True:
        key = False

        # Capture frame-by-frame
        img = cam.read()
        img = np.flip(img, axis=1)  # Flip the image horizontally

        # Reduce the resolution to improve performance (e.g., 320x240)
        img = imutils.resize(img, width=320, height=240)

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply a smaller Gaussian blur to reduce processing load
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)  # Smaller kernel size

        # Define the color range for detecting the steering wheel
        colourLower = np.array([53, 55, 209])
        colourUpper = np.array([180, 255, 255])

        height = img.shape[0]
        width = img.shape[1]

        # Create a mask based on the color range
        mask = cv2.inRange(blurred, colourLower, colourUpper)

        # Apply morphological operations to clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Define regions of interest (ROI) for detecting movement
        upContour = mask[0:height//2, 0:width]
        downContour = mask[3*height//4:height, 2*width//5:3*width//5]

        # Find contours in the ROI
        cnts_up = cv2.findContours(upContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_up = imutils.grab_contours(cnts_up)

        cnts_down = cv2.findContours(downContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_down = imutils.grab_contours(cnts_down)

        # Process the upper contour (steering control)
        if len(cnts_up) > 0:
            c = max(cnts_up, key=cv2.contourArea)
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + 0.000001))

            if cX < (width // 2 - 35):
                PressKey(A)
                key = True
                currentKey.append(A)
            elif cX > (width // 2 + 35):
                PressKey(D)
                key = True
                currentKey.append(D)

        # Process the lower contour (nitro control)
        if len(cnts_down) > 0:
            PressKey(Space)
            key = True
            currentKey.append(Space)

        # Draw rectangles and labels for the control areas
        img = cv2.rectangle(img, (0, 0), (width//2-35, height//2), (0, 255, 0), 1)
        cv2.putText(img, 'LEFT', (110, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

        img = cv2.rectangle(img, (width//2 + 35, 0), (width-2, height//2), (0, 255, 0), 1)
        cv2.putText(img, 'RIGHT', (440, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

        img = cv2.rectangle(img, (2*(width//5), 3*(height//4)), (3*width//5, height), (0, 255, 0), 1)
        cv2.putText(img, 'NITRO', (2*(width//5) + 20, height-10), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

        # Display the resulting frame
        cv2.imshow("Steering", img)

        # Release keys if no key was pressed in this frame
        if not key and len(currentKey) != 0:
            for current in currentKey:                                                                                                                                           
                ReleaseKey(current)
            currentKey = list()

        # Exit the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Clean up
    cam.stop()
    cv2.destroyAllWindows()
