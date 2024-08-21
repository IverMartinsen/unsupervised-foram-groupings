import yolov5
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import get_boxes_yolo, compute_iou


path_to_slide = "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_ zoom 6.25.JPG"
slide = np.array(Image.open(path_to_slide)).astype(np.uint8)

model = yolov5.load("keremberke/yolov5m-aerial-sheep")  # pretrained model for sheep detection

# ============ get bounding boxes ============
boxes = []
num_boxes = []

for size in [1024, 2048, 4096]:
    boxes_ = get_boxes_yolo(slide, size=size)  # get bounding boxes using the sheep detector
    num_boxes.append(len(boxes_))
    boxes += boxes_

iou = compute_iou(boxes) * (np.eye(len(boxes)) == 0)
iou[:num_boxes[0], :num_boxes[0]] = 0
iou[num_boxes[0]:num_boxes[0] + num_boxes[1], num_boxes[0]:num_boxes[0] + num_boxes[1]] = 0
iou[num_boxes[0] + num_boxes[1]:, num_boxes[0] + num_boxes[1]:] = 0
iou = np.triu(iou)

boxes = [box for i, box in enumerate(boxes) if np.all(iou[i] < 0.1)]
# ============ get bounding boxes (end) ============

# ============ display the bounding boxes ============
fig, ax = plt.subplots()
plt.imshow(slide)
for box in boxes:
    rect = plt.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], edgecolor="r", facecolor="none")
    ax.add_patch(rect)
plt.axis("off")
plt.savefig(f"thumbnail.png", dpi=300)
# ============ display the bounding boxes (end) ============

# ============ save the cropped images ============
thickness = 10
for box in boxes:
    box = [int(x) for x in box]
    
    box_id = f"{box[0]}_{box[1]}_{box[2]}_{box[3]}"
    slide_with_box = slide.copy()
    slide_with_box[box[0]:box[2], box[1] - thickness:box[1] + thickness] = [255, 0, 0]
    slide_with_box[box[0]:box[2], box[3] - thickness:box[3] + thickness] = [255, 0, 0]
    slide_with_box[box[0] - thickness:box[0] + thickness, box[1]:box[3]] = [255, 0, 0]
    slide_with_box[box[2] - thickness:box[2] + thickness, box[1]:box[3]] = [255, 0, 0]
    
    # resize the image
    slide_with_box = Image.fromarray(slide_with_box).resize((640, 480))
    slide_with_box = np.array(slide_with_box)    
    plt.imsave(f"Data/BOXES/box_{box_id}.png", slide_with_box, dpi=50)
    
    # save the fossil
    fossil = slide[box[0]:box[2], box[1]:box[3]]
    plt.imsave(f"Data/CROPS/fossil_{box_id}.png", fossil)
# ============ save the cropped images (end) ============


