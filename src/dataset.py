import os
import requests
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO
import json

ann_file = 'dataset/annotations/captions_val2017.json'
coco = COCO(ann_file)

# loading 500 coco images
img_ids = coco.getImgIds()[:500]
images_data = coco.loadImgs(img_ids)

print(f"downloading 500 images...")
dataset_subset = []
img_captions = {}
for img_info in images_data:
    # getting the image
    response = requests.get(img_info['coco_url'])
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_path = f"dataset/{img_info['file_name']}"
    img.save(img_path)

    # getting the captions
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)
    captions = [ann['caption'] for ann in anns]
    img_captions[img_info['file_name']] = {
        "captions": captions,
        "local_path": img_path
    }

with open('dataset/img_captions.json', 'w') as f:
    json.dump(img_captions, f, indent=4)

print("done! dataset ready")