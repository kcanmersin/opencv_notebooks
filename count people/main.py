import cv2
import time
import numpy as np
import os
from ultralytics import YOLO

video_path="Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4"
line1 = []
line2 = []
line3 = []

def mouse_callback(event, x, y, flags, param):
    global line1, line2, line3
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line1) < 2:
            line1.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  
        elif len(line2) < 2:
            line2.append((x, y))
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1) 
        elif len(line3) < 2:
            line3.append((x, y))
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1) 

cv2.namedWindow("Select Lines")
cv2.setMouseCallback("Select Lines", mouse_callback)

kamera = cv2.VideoCapture(video_path)
ret, frame = kamera.read()
if not ret:
    print("Start frame cannot taken")
    exit()

#Select lines
print("First select middle line points, then select upper line points, then select bottom line points")
while True:
    text1 = "Select middle line points"
    text2 = "Select upper line points"
    text3 = "Select bottom line points"
    text4 = "Press q to continue"  # Added this line
    
    # Draw the frame
    display_frame = frame.copy()
    
    # Add text to the frame
    cv2.putText(display_frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if len(line1) >= 2:
        cv2.putText(display_frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if len(line2) >= 2:
        cv2.putText(display_frame, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add the "Press q to continue" text to the frame
    cv2.putText(display_frame, text4, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Select Lines", display_frame)
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model = YOLO("yolov8x.pt")

font = cv2.FONT_HERSHEY_DUPLEX

prev_frame_time = 0
new_frame_time = 0

line1 = np.array(line1, dtype=np.int32)
line1 = line1.reshape((-1, 1, 2))

line2 = np.array(line2, dtype=np.int32)
line2 = line2.reshape((-1, 1, 2))

line3 = np.array(line3, dtype=np.int32)
line3 = line3.reshape((-1, 1, 2))


region1=np.array([line1[0],line2[0],line2[1],line1[1]])
region1 = region1.reshape((-1,1,2))

region2=np.array([line1[0],line3[0],line3[1],line1[1]])
region2 = region2.reshape((-1,1,2))

#middle line points
pt1 = tuple(line1[0][0])  
pt2 = tuple(line1[1][0])  


upper_id=set()
bottom_id=set()

in_id=set()
out_id=set()
frame_height, frame_width, _ = frame.shape

while True:

    ret, frame = kamera.read()
    
    if not ret:
        break
    
    # convert to rgb
    rgb_img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   
    #Draw middle line
    cv2.line(frame,  pt1, pt2,  (255,0,0), 3)

    results = model.track(rgb_img, persist=True, verbose=False)
    
 
    for i in range(len(results[0].boxes)):
        #x1 y1 left top corner. x2 y2 right bottom corner coordinates.
        x1,y1,x2,y2=results[0].boxes.xyxy[i]
        score=results[0].boxes.conf[i]
        cls=results[0].boxes.cls[i]

        # ids of the objects
        ids=results[0].boxes.id[i]
        
        x1,y1,x2,y2,score,cls,ids=int(x1),int(y1),int(x2),int(y2),float(score),int(cls),int(ids)
        
        if score<0.1:
            continue
        # if object is not a person continue
        if cls!=0:
            continue

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        
        # middle point of objects
        cx=int(x1/2+x2/2)
        cy=int(y1/2+y2/2)
        cv2.circle(frame,(cx,cy),4,(0,255,255),-1)
        
        #  check object is in the region1 or not
        inside_region1=cv2.pointPolygonTest(region1,(cx,cy),False)
        
      
        if inside_region1>0:
            if ids in bottom_id:
                # Add object id to the list of people who left the region1
                out_id.add(ids)
                cv2.line(frame,  pt1, pt2, (255,255,255), 3)

            upper_id.add(ids)
            
        inside_region2=cv2.pointPolygonTest(region2,(cx,cy),False)
        if inside_region2>0:
            if ids in upper_id:
                cv2.line(frame,  pt1, pt2,  (255,255,255), 3)
                in_id.add(ids)
            bottom_id.add(ids)
        
    
 
    in_id_str='IN: '+str(len(in_id))
    out_id_str='OUT: '+str(len(out_id))
    

    frame[0:40, 0:120] = (153, 0, 102)

# Sağ üst köşeye mavi bir alan ekleyin
    frame[0:40, frame_width - 120:] = (153, 0, 102) 

    cv2.putText(frame, in_id_str,(0, 30), font, 1, (255,255,255), 1)
    cv2.putText(frame, out_id_str, (frame_width - 120, 30), font, 1, (255,255,255), 1)
    
    cv2.polylines(frame,[region1],True,(255,0,0),2)
    cv2.polylines(frame,[region2],True,(255,0,255),2)
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()