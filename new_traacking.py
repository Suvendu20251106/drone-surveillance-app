import cv2
import numpy as np
from geopy.distance import distance
from geopy import Point

import geocoder

def get_current_location():
    g = geocoder.ip('me')
    return g.latlng if g.ok else (None, None)


net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 800, 800)

KNOWN_WIDTH = 7  
FOCAL_LENGTH = 3891.43  

camera_lat, camera_lon = get_current_location()
if camera_lat is None or camera_lon is None:
    print("Failed to get current location!")
    exit()

camera_heading = 90.0

def calculate_distance(knownWidth, focalLength, perWidth):
    if perWidth == 0:
        return -1
    return (knownWidth * focalLength) / perWidth

def estimate_object_gps(distance_to_object, bearing):
    start_point = Point(camera_lat, camera_lon)
    d = distance(meters=distance_to_object)
    destination = d.destination(point=start_point, bearing=bearing)
    return destination.latitude, destination.longitude

tracker = None
tracking = False
track_box = None
frame_copy = None
click_x = click_y = -1

def click_event(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y

cv2.setMouseCallback("Object Detection", click_event)

def create_tracker():
    return cv2.TrackerCSRT_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()
    h, w = frame.shape[:2]

    if tracking and tracker:
        success, box = tracker.update(frame)
        if success:
            x, y, w_box, h_box = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            cv2.putText(frame, "Locked Target", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            tracking = False
            tracker = None


    blob = cv2.dnn.blobFromImage(cv2.resize(frame_copy, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = classes[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            boxes.append((startX, startY, endX, endY, label))

            
            obj_center_x = (startX + endX) // 2

        
            position = "Center"
            if obj_center_x < w // 3:
                position = "Left"
            elif obj_center_x > 2 * w // 3:
                position = "Right"

            
            box_width = endX - startX
            distance_cm = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, box_width)
            distance_m = distance_cm / 100.0

            
            fov = 60 
            pixel_offset = obj_center_x - (w // 2)
            bearing_offset = (pixel_offset / w) * fov
            true_bearing = (camera_heading + bearing_offset) % 360

            
            obj_lat, obj_lon = estimate_object_gps(distance_m, true_bearing)

            label_text = f"{label}: {confidence:.2f} | {position} | Dist: {int(distance_cm)}cm"
            gps_text = f"GPS: {obj_lat:.5f}, {obj_lon:.5f}"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (startX, startY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, gps_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if click_x != -1 and click_y != -1:
        for startX, startY, endX, endY, label in boxes:
            if startX <= click_x <= endX and startY <= click_y <= endY:
                bbox = (startX, startY, endX - startX, endY - startY)
                tracker = create_tracker()
                tracker.init(frame, bbox)
                tracking = True
                print(f"Locked on: {label}")
                break
        click_x = click_y = -1

    cv2.imshow("Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracker = None
        tracking = False
        print("Tracking reset.")

cap.release()
cv2.destroyAllWindows()
