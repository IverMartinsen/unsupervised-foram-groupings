import os
import yolov5
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import get_boxes_yolo, compute_iou

path_to_slides=[
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (1).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (2).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (3).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (4).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (5).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (6).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (7).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (8).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (9).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (10).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (11).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (12).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (13).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (14).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (15).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (16).JPG",
    "Data/Gol-F-30-3_19-20/Gol-F-30-3, 19-20_zoom 35 (17).JPG",
    ]

zoom_level = 35

fname = "Gol-F-30-3, 19-20_zoom 35"

for path_to_slide in path_to_slides:

    slide = np.array(Image.open(path_to_slide)).astype(np.uint8)

    model = yolov5.load("keremberke/yolov5m-aerial-sheep")  # pretrained model for sheep detection

    # ============ get bounding boxes ============
    boxes = []
    num_boxes = []

    scales = {6.25: [1024, 2048, 4096], 16: [1024, 2048], 35: [512, 1024]}  # scales for different zoom levels 

    # Large objects use 1024, 2048 for medium and 4096 for small objects
    for size in scales[zoom_level]:
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
    dst = os.path.basename(path_to_slide.strip(".JPG"))
    
    fig, ax = plt.subplots()
    plt.imshow(slide)
    for box in boxes:
        rect = plt.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    plt.axis("off")
    plt.savefig(f"Data/{dst}_thumbnail.png", dpi=300)
    # ============ display the bounding boxes (end) ============

    # ============ save the cropped images ============
    thickness = 10

    BOXES = "Data/BOXES_" + fname + "/images"
    CROPS = "Data/CROPS_" + fname + "/images"
    os.makedirs(BOXES, exist_ok=True)
    os.makedirs(CROPS, exist_ok=True)

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
        plt.imsave(f"{BOXES}/box_{box_id}.png", slide_with_box, dpi=50)
        
        # save the fossil
        #fossil = slide[box[0]:box[2], box[1]:box[3]]
        #plt.imsave(f"{CROPS}/fossil_{box_id}.png", fossil)
        fossil = Image.fromarray(slide).crop((box[1], box[0], box[3], box[2]))
        fossil.save(f"{CROPS}/fossil_{box_id}.jpg")
    # ============ save the cropped images (end) ============


