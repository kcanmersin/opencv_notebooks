import cv2


face_cascade = cv2.CascadeClassifier(r"C:\Users\KC\Desktop\lip to text\haarcascade_frontalface_default.xml")

mouth_cascade = cv2.CascadeClassifier(r"C:\Users\KC\Desktop\lip to text\haarcascade_mcs_mouth.xml")

path="bbas1s.mp4"
cap = cv2.VideoCapture(path)

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
   
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # make grey only face
        roi_gray = gray[y:y+h, x:x+w]
        
        #detect mouth
        mouths = mouth_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,  
            minNeighbors=50, 
            minSize=(20, 20) 
        )

        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(frame, (x+mx, y+my), (x+mx+mw, y+my+mh), (0, 0, 255), 2)
            
            cv2.putText(frame, f'Point 1 ({x}, {y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f'Point 2 ({x+w}, {y})', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f'Point 3 ({x}, {y+h})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f'Point 4 ({x+w}, {y+h})', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
