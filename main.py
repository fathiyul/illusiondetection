import cv2
import numpy as np
import sys

bin_thresh = 1 

threshold_area = 100

color = (0, 255, 255)

cap = cv2.VideoCapture(sys.argv[1]) 

ret, first_frame = cap.read()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

bitrate = frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

w, h, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "output.mp4"
output_video = cv2.VideoWriter(output_path, fourcc, fps, ((int)(w), int(h)))

while(cap.isOpened()): 

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_diff = gray-prev_gray

    if frame_diff.mean() < 1.0:
        continue
    
    cv2.imshow("input", frame)
    
    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mbg_bin = cv2.threshold(magnitude,bin_thresh,255,cv2.THRESH_BINARY)

    contour, _ = cv2.findContours(mbg_bin,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    cnt_count = 0
    for cnt in contour:
        area = cv2.contourArea(cnt)
        cnt_xywh = cv2.boundingRect(cnt)
        is_cnt_small = cnt_xywh[2] < w and cnt_xywh[3] < h
        if area > threshold_area and is_cnt_small:
            cv2.drawContours(frame,[cnt],-1,color,3)
    
    cv2.imshow("flow magnitude", magnitude)
    cv2.imshow("processed frame", frame)

    output_video.write(frame)
    
    # Updates previous frame 
    prev_gray = gray
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()
output_video.release()