import cv2
import numpy as np
import math

from object_detection import ObjectDetection

MIN_DISTANCE = 20
VIDEO_FILE = "video.mp4"

class ObjectTracker:
    def __init__(self):
        self.track_id = 0
        self.tracking_objects = {}

    def track_objects(self, frame, center_points_cur_frame):
        tracking_objects_copy = self.tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < MIN_DISTANCE:
                    self.tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            if not object_exists:
                self.tracking_objects.pop(object_id)

        for pt in center_points_cur_frame:
            self.tracking_objects[self.track_id] = pt
            self.track_id += 1

    def draw_tracked_objects(self, frame):
        for object_id, pt in self.tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

def main():
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    od = ObjectDetection()
    object_tracker = ObjectTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center_points_cur_frame = []

        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        object_tracker.track_objects(frame, center_points_cur_frame)
        object_tracker.draw_tracked_objects(frame)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
