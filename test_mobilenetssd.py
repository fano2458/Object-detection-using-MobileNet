from src.model_mobile import Detector
import torch
import cv2
from src.utils import non_max_suppression, cellboxes_to_boxes


weights_path = "weights/ssd_lite_43.pht"
checkpoint = torch.load(weights_path)

model = Detector().to("cuda")
model.load_state_dict(checkpoint)
model.eval()

cap = cv2.VideoCapture(0) 
count = 0

while True:
    ret, frame = cap.read()

    # Resize and convert BGR to RGB
    image = cv2.resize(frame, (320, 320))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 255.0

    with torch.no_grad():
        bboxes = cellboxes_to_boxes(model(image))
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

    # print(len(bboxes))
    try:
        cls, conf, x, y, w, h = bboxes[0]

        img_h, img_w, img_ch = frame.shape

        x_img = x * img_w
        y_img = y * img_h
        w_img = w * img_w
        h_img = h * img_h

        tl_x = int(x_img - w_img // 2)
        tl_y = int(y_img - h_img // 2)
        br_x = int(x_img + w_img // 2)
        br_y = int(y_img + h_img // 2)

        cv2.rectangle(frame, (tl_x, tl_y), (br_x, br_y), (255, 0, 255), 2, -1)
    except IndexError:
        print("no detections")

    cv2.imwrite(f'output/frame_{count}.png', frame)

    if count > 10:
        break

    count += 1
