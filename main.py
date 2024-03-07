import cv2
import imutils
import numpy as np
import logging

# Initialize logging to output to both the console and a file for better monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Create and configure a file handler to save log messages to a file
fh = logging.FileHandler('app.log', mode='w')
fh.setLevel(logging.INFO)
# Create and configure a console handler to print log messages
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Define a formatter to structure log messages uniformly
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# Add the file and console handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# Define HSV color bounds for detecting specific colors in video frames
color_bounds = {
    'blue': {'lower': np.array([100, 50, 50]), 'upper': np.array([130, 255, 255])},
    'red': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
    'green': {'lower': np.array([40, 50, 50]), 'upper': np.array([80, 255, 255])},
    'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
    'orange': {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
    'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 30])},
    'white': {'lower': np.array([0, 0, 200]), 'upper': np.array([180, 30, 255])}
}


def get_min_contour_area(frame_width, frame_height):
    """
    Calculate the minimum contour area based on the frame dimensions to adapt dynamically
    to different video resolutions.
    """
    return (frame_width * frame_height) / 1000


def process_frame(frame, color_bounds, chosen_color, min_contour_area):
    """
    Apply Gaussian blur to the frame, convert it to the HSV color space, create a mask based on
    specified color bounds, and find contours within this mask.
    """
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, color_bounds[chosen_color]["lower"], color_bounds[chosen_color]["upper"])
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return frame, cnts


def draw_contours(frame, cnts, min_contour_area):
    """
    Iterate through each contour detected and draw it on the frame if it is above the minimum
    area threshold. Also, log the center coordinates of significant contours.
    """
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_contour_area:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 5)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                logger.info(f"Center coordinates: {center}")
    return frame


def main():
    """
    Main function to set up video capture, process and display video frames in a loop,
    and handle user input to quit. Processes frames to detect and annotate specified
    color objects and logs their information.
    """
    chosen_color = 'blue'  # Change this based on the desired color to detect
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        # Resize frame for faster processing and get dynamic contour area
        frame = cv2.resize(frame, (640, 480))
        min_contour_area = get_min_contour_area(frame.shape[1], frame.shape[0])

        # Process the frame and draw contours
        frame, cnts = process_frame(frame, color_bounds, chosen_color, min_contour_area)
        frame = draw_contours(frame, cnts, min_contour_area)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up: release camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
