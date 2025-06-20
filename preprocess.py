import os
import json
import shutil
from PIL import Image, ImageDraw
import numpy as np

IDD_PATH = "data/idd_segmentation"
ROBOFLOW_PATH = "data/pothole-detection-system-1"
OUT_PATH = "data/processed"

POTHOLE_CLASS_ID = 27  # Unique class ID for pothole

# You can adjust this if you only care about a subset
IDD_LABELS = {
    "road": 0,
    "drivable fallback": 1,
    "sidewalk": 2,
    "non-drivable fallback": 3,
    "person": 4,
    "rider": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "autorickshaw": 8,
    "car": 9,
    "truck": 10,
    "bus": 11,
    "vehicle fallback": 12,
    "curb": 13,
    "wall": 14,
    "fence": 15,
    "guard rail": 16,
    "billboard": 17,
    "traffic sign": 18,
    "traffic light": 19,
    "pole": 20,
    "obs-str-bar-fallback": 21,
    "building": 22,
    "bridge/tunnel": 23,
    "vegetation": 24,
    "sky": 25,
    "fallback background": 26
}
NAME2ID = IDD_LABELS

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_polygons(json_file, img_size):
    mask = Image.new('L', img_size, 255)  # default unknown class
    draw = ImageDraw.Draw(mask)

    with open(json_file, 'r') as f:
        data = json.load(f)
        for obj in data['objects']:
            label = obj['label']
            if label not in NAME2ID:
                continue
            polygon = obj['polygon']
            if len(polygon) >= 3:
                draw.polygon(polygon, fill=NAME2ID[label])

    return mask

def process_idd():
    print("🚗 Generating masks from IDD JSON annotations...")

    for split in ["train", "val"]:
        left_img_root = os.path.join(IDD_PATH, "leftImg8bit", split)
        gt_json_root = os.path.join(IDD_PATH, "gtFine", split)
        out_img_dir = os.path.join(OUT_PATH, split, "images")
        out_mask_dir = os.path.join(OUT_PATH, split, "masks")

        ensure_dir(out_img_dir)
        ensure_dir(out_mask_dir)

        folders = os.listdir(left_img_root)
        for folder in folders:
            img_folder = os.path.join(left_img_root, folder)
            ann_folder = os.path.join(gt_json_root, folder)

            if not os.path.isdir(img_folder):
                continue

            img_files = [f for f in os.listdir(img_folder) if f.endswith("_leftImg8bit.png")]
            for img_file in img_files:
                base_name = img_file.replace("_leftImg8bit.png", "")
                img_path = os.path.join(img_folder, img_file)
                json_path = os.path.join(ann_folder, base_name + "_gtFine_polygons.json")

                if not os.path.exists(json_path):
                    print(f"⚠ Missing annotation: {json_path}")
                    continue

                img = Image.open(img_path)
                mask = parse_polygons(json_path, img.size)

                # Save image and mask
                out_img_name = f"{folder}_{img_file}"
                out_mask_name = f"{folder}_{base_name}_mask.png"

                img.save(os.path.join(out_img_dir, out_img_name))
                mask.save(os.path.join(out_mask_dir, out_mask_name))

    print("✅ IDD JSON → Mask conversion done.")

def convert_coco_to_masks(json_path, image_dir, out_split):
    """
    Convert COCO format annotations to per-image segmentation masks.
    For pothole dataset.
    """
    print(f"🕳 Processing Roboflow Pothole dataset {out_split} split...")

    out_img_dir = os.path.join(OUT_PATH, out_split, "images")
    out_mask_dir = os.path.join(OUT_PATH, out_split, "masks")

    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)

    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Build image id to filename mapping
    images = {img['id']: img['file_name'] for img in coco['images']}
    annotations = coco['annotations']

    # Create blank masks for each image
    masks = {}

    # Since Roboflow's segmentation annotations can be polygons,
    # use pycocotools or manual rasterization
    # But here we'll do simple bounding box mask since
    # pycocotools may not be installed (you can add if needed)

    from PIL import ImageDraw

    # Initialize blank masks
    for img_id, file_name in images.items():
        # Load image to get size
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path)
        mask = Image.new('L', img.size, 0)  # Single channel mask
        masks[img_id] = mask

    # Draw polygons on mask for each annotation of pothole class
    for ann in annotations:
        if ann['category_id'] != 1:  # Assuming 1 = pothole class in Roboflow
            continue
        img_id = ann['image_id']
        seg = ann['segmentation']  # Polygon list(s)

        mask = masks[img_id]
        draw = ImageDraw.Draw(mask)

        # segmentation can be list of polygons
        if isinstance(seg[0], list):
            polygons = seg
        else:
            polygons = [seg]

        for polygon in polygons:
            xy = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
            draw.polygon(xy, fill=POTHOLE_CLASS_ID)

    # Save masks and copy images
    for img_id, file_name in images.items():
        # Save mask
        mask = masks[img_id]
        mask_name = file_name.rsplit('.', 1)[0] + "_mask.png"
        mask.save(os.path.join(out_mask_dir, mask_name))

        # Copy image to output folder
        src_img_path = os.path.join(image_dir, file_name)
        dst_img_path = os.path.join(out_img_dir, file_name)
        shutil.copy(src_img_path, dst_img_path)

    print(f"✅ Roboflow Pothole dataset {out_split} processed.")

if __name__ == "__main__":
    process_idd()
    convert_coco_to_masks(
        json_path=os.path.join(ROBOFLOW_PATH, "train", "_annotations.coco.json"),
        image_dir=os.path.join(ROBOFLOW_PATH, "train"),
        out_split="train"
    )
    convert_coco_to_masks(
        json_path=os.path.join(ROBOFLOW_PATH, "valid", "_annotations.coco.json"),
        image_dir=os.path.join(ROBOFLOW_PATH, "valid"),
        out_split="val"
    )