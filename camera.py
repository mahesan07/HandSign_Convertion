import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()     # The cap.read() function is used to capture a frame from the video stream.
                                # It returns two values: ret, which is a boolean indicating whether the frame was successfully captured, and frame,
                                # which contains the actual image data of the captured frame.
    if not ret:
        print("Cannot receive frame")
        break

    cv2.imshow("Sign Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()