import datetime
import base64
import torch
import cv2
import numpy as np
import socketio
import eventlet.wsgi

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from flask import Flask
from PIL import Image

from model import ResNetAutoSteer

# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

velocity = 0
control_outputs = [0, 0, 0]

weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model_rcnn = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=0.7)
model_rcnn.to("cuda")
model_rcnn.eval()

model_steer = ResNetAutoSteer()
model_steer.load_state_dict(torch.load('models/ResNetSteer_v5_2.pth'))
model_steer.eval()
model_steer.to("cuda")


def ts():
    current_time = datetime.datetime.now()
    hours = current_time.hour
    minutes = current_time.minute
    seconds = current_time.second
    milliseconds = current_time.microsecond // 1000
    return f"\r{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03} - "


def check_for_car(pred):
    detections = []
    stop = 0
    slow = 0
    car_idx = []
    for element, label in enumerate(pred["labels"]):
        if label.item() in [2, 3, 4, 6, 7, 8, 9]:
            car_idx.append(element)
    for element, corners in enumerate(pred["boxes"]):
        if element in car_idx:
            center = [(corners[0].item() + corners[2].item()) / 2, (corners[1].item() + corners[3].item()) / 2]
            width = corners[2].item() - corners[0].item()
            detections.append([center, width])

    for car in detections:

        # STOP
        if 320 < car[0][0] < 360 and 500 > car[1] > 75:
            stop += 1
            print(ts() + "Status: STOP (Left)", end="")
            continue
        if 500 < car[0][0] < 600 and 500 > car[1] > 50:
            stop += 1
            print(ts() + "Status: STOP (Center)", end="")
            continue
        if 740 < car[0][0] < 840 and 500 > car[1] > 75:
            stop += 1
            print(ts() + "Status: STOP (Right)", end="")
            continue

        # SLOW
        if 320 < car[0][0] < 360 and 500 > car[1] > 60:
            slow += 1
            print(ts() + "Status: SLOW (Left)", end="")
            continue
        if 500 < car[0][0] < 600 and 500 > car[1] > 40:
            slow += 1
            print(ts() + "Status: SLOW (Center)", end="")
            continue
        if 740 < car[0][0] < 840 and 500 > car[1] > 55:
            slow += 1
            print(ts() + "Status: SLOW (Right)", end="")
            continue

    return stop > 0, slow > 0


def adapt_steer(network_steer):
    steer = (network_steer + 0.015) * 0.8
    if 0.25 > abs(steer) > 0.20:
        steer = steer * 1.7
    elif 0.20 > abs(steer) > 0.15:
        steer = steer * 1.5
    elif 0.15 > abs(steer) > 0.03:
        steer = steer * 1.3
    elif 0.03 > abs(steer):
        steer = steer * 0.3

    steer = ((velocity + 10) / 30) * steer
    return steer


def control(b_stop, b_slow, v_steer):
    if b_stop:
        if velocity > 15:
            return [v_steer, 0, 1]
        elif velocity > 10:
            return [v_steer, 0, 0.5]
        elif velocity > 1:
            return [v_steer, 0, 0.1]
        else:
            return [v_steer, 0, 0]

    elif b_slow:
        if velocity > 15:
            return [v_steer, 0, 0.4]
        elif velocity > 10:
            return [v_steer, 0, 0.2]
        else:
            return [v_steer, 0, 0]
    else:
        if velocity > 20:
            print(ts() + "Status: GO (Keep velocity)", end="")
            return [v_steer, 0, 0]
        elif velocity > 15:
            print(ts() + "Status: GO (Keep velocity)", end="")
            return [v_steer, 0.04, 0]
        else:
            print(ts() + "Status: GO (Accelerate)", end="")
            return [v_steer, 0.35, 0]


@sio.on("send_image")
def on_image(sid, data):
    global control_outputs

    img_data = data["image"]
    img_bytes = base64.b64decode(img_data)
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_inp = Image.fromarray(rgb_img)

        with torch.no_grad():

            preprocess = weights.transforms()
            batch = [preprocess(img_inp).to("cuda")]
            prediction = model_rcnn(batch)[0]
            stop, slow = check_for_car(prediction)

            img_tensor = torch.autograd.Variable(torch.FloatTensor(img))
            img_tensor = img_tensor.to("cuda", non_blocking=True)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            output = model_steer(img_tensor.float())
            steer = adapt_steer(output.item())
            control_outputs = control(stop, slow, steer)

        cv2.namedWindow("image from unity", cv2.WINDOW_NORMAL)
        cv2.imshow("image from unity", img)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return
    else:
        print("Invalid image data")


# listen for the event "vehicle_data"
@sio.on("vehicle_data")
def vehicle_command(sid, data):
    global velocity
    velocity = float(data["velocity"].replace(',', '.'))

    if data:
        send_control(*control_outputs)
    else:
        print("data is empty")


@sio.event
def connect(sid, environ):
    print("Client connected")
    print("\n")
    send_control(0, 0, 0)


def send_control(steering_angle, throttle, brake):
    sio.emit(
        "control_command",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
            "brake": brake.__str__(),
        },
        skip_sid=True,
    )


@sio.event
def disconnect(sid):
    print("Client disconnected")


app = socketio.Middleware(sio, app)
# Connect to Socket.IO client
if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
