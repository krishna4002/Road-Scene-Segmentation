from PIL import Image
import numpy as np

MERGED_CLASS_NAMES = [
    "road", "drivable fallback", "sidewalk", "non-drivable fallback", "person", "rider",
    "motorcycle", "bicycle", "autorickshaw", "car", "truck", "bus", "fallback vehicle",
    "curb", "wall", "fence", "guard rail", "billboard", "traffic sign", "traffic light",
    "pole", "obs", "building", "bridge", "vegetation", "sky", "unknown", "pothole"
]

MERGED_COLORS = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Green
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (128, 0, 0),       # Maroon
    (0, 128, 0),       # Dark Green
    (0, 0, 128),       # Navy
    (128, 128, 0),     # Olive
    (128, 0, 128),     # Purple
    (0, 128, 128),     # Teal
    (255, 165, 0),     # Orange
    (0, 100, 0),       # Dark Green Variant
    (139, 0, 139),     # Dark Magenta
    (75, 0, 130),      # Indigo
    (255, 20, 147),    # Deep Pink
    (0, 191, 255),     # Deep Sky Blue
    (210, 105, 30),    # Chocolate
    (124, 252, 0),     # Lawn Green
    (47, 79, 79),      # Dark Slate Gray
    (220, 20, 60),     # Crimson
    (255, 215, 0),     # Gold
    (106, 90, 205),    # Slate Blue
    (0, 250, 154),     # Medium Spring Green
    (199, 21, 133),    # Medium Violet Red
    (135, 206, 235),   # Sky Blue
    (0, 0, 0)          # Black (for pothole or unknown class)
]

def decode_merged_segmap(label_mask):
    r = np.zeros_like(label_mask).astype(np.uint8)
    g = np.zeros_like(label_mask).astype(np.uint8)
    b = np.zeros_like(label_mask).astype(np.uint8)
    
    for i, color in enumerate(MERGED_COLORS):
        r[label_mask == i] = color[0]
        g[label_mask == i] = color[1]
        b[label_mask == i] = color[2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb