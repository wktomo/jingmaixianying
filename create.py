import cv2
import numpy as np
out = cv2.VideoWriter('test_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
for i in range(100):
    frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    out.write(frame)
out.release()