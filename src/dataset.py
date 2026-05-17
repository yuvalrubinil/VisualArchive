import requests
from pycocotools.coco import COCO
from PIL import Image
from io import BytesIO

N = 500

ann_file = 'dataset/annotations/captions_val2017.json'
coco = COCO(ann_file)

# loading N coco images
img_ids = coco.getImgIds()[:N]
images_data = coco.loadImgs(img_ids)

print(f"downloading {N} images...")
dataset_subset = []
for img_info in images_data:
    response = requests.get(img_info['coco_url'])
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_path = f"dataset/{img_info['file_name']}"
    img.save(img_path)

print("done! dataset ready")
