import glob
from PIL import Image
import cv2

# filepaths
fp_in = "runs/detect/exp2/test_video.mp4"
fp_out = "gallery/demo.gif"

cap = cv2.VideoCapture(fp_in)

total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("total number of frames:", total_frame)
imgs = []
counter = 0
STEP = 5
while(counter < total_frame):
    ret, frame = cap.read()
    if ret:
        if counter % STEP == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(frame))
            print("images appended:", len(imgs))
    counter += 1

print("creating GIF now...")
imgs[0].save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
