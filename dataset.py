import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class RoadSceneDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(512, 512)):
        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.transform = transform
        self.target_size = target_size

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

        self.masks = sorted([
            f for f in os.listdir(self.mask_dir)
            if f.endswith('.png')
        ])

        assert len(self.images) == len(self.masks), "Image and mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB").resize(self.target_size)
        mask = Image.open(mask_path).resize(self.target_size, resample=Image.NEAREST)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        mask = T.PILToTensor()(mask).squeeze(0).long()

        return image, mask