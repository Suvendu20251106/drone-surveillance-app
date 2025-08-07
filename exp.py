import cv2

KNOWN_DISTANCE = 60  # cm
KNOWN_WIDTH = 7      # cm

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image.")
    exit()

# Manually draw a bounding box around the object using OpenCV GUI
bbox = cv2.selectROI("Calibration", frame, fromCenter=False, showCrosshair=True)
startX, startY, w_box, h_box = bbox
endX = startX + w_box

perceived_width = w_box  # in pixels

focal_length = (perceived_width * KNOWN_DISTANCE) / KNOWN_WIDTH
print(f"Calibrated focal length = {focal_length:.2f} pixels")

cv2.destroyAllWindows()
