import cv2
import torch
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
import threading
import time
import queue

DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(
    f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

depth_queue = queue.Queue()


def measure_depth_thread(frame):
    depth = model.infer_image(frame)
    depth_queue.put(depth)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_depth_measure_time = time.time()
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("Normal Video", frame)

    # Measure Depth after every 1 second. Can be reduced as per the computation power.
    current_time = time.time()
    if current_time - last_depth_measure_time >= 1:
        depth_thread = threading.Thread(
            target=measure_depth_thread, args=(frame,))
        depth_thread.start()
        last_depth_measure_time = current_time

    if not depth_queue.empty():
        depth = depth_queue.get()
        plt.imshow(depth, cmap='plasma')
        plt.title("Depth Estimate")
        plt.pause(0.001)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
