"""
Data loader for DINOv3 Triplet Model training
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def make_dinov3_style_augment(global_crops_size: int = 224):
    """
    Create DINOv3-style strong augmentation pipeline.

    Note: Normalization and ToTensor are handled by AutoImageProcessor.

    Args:
        global_crops_size (int): Size of the crop

    Returns:
        torchvision.transforms.Compose: Composed augmentation transforms
    """
    return T.Compose([
        T.RandomResizedCrop(
            size=global_crops_size,
            scale=(0.3, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        ),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3),
    ])


class DinoV3TwoViewTripletDataset(Dataset):
    """
    Dataset that returns two augmented views from each image for DINOv3 Triplet training.
    """

    def __init__(self, root_dir, label_map, processor, augment=None):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory containing class subdirectories (e.g., ./data/train)
            label_map (dict): Mapping from class name to label ID {"classA": 0, "classB": 1, ...}
            processor: AutoImageProcessor for DINOv3
            augment: DINOv3-style augmentation transform (PIL -> PIL)
        """
        self.samples = []  # List of (path, label) tuples
        self.label_map = label_map
        self.processor = processor
        self.augment = augment or make_dinov3_style_augment()

        # Collect image paths
        for class_name, label_id in label_map.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample with two augmented views.

        Args:
            idx (int): Index

        Returns:
            dict: Dictionary containing:
                - "pixel_values1": [C, H, W] tensor (anchor view)
                - "pixel_values2": [C, H, W] tensor (positive view)
                - "labels": label ID
        """
        path, label = self.samples[idx]

        # Load image
        img = Image.open(path).convert("RGB")

        # Generate two augmented views from the same image
        view1 = self.augment(img)
        view2 = self.augment(img)

        # Process views with AutoImageProcessor (tensor conversion and normalization)
        inputs1 = self.processor(images=view1, return_tensors="pt")
        inputs2 = self.processor(images=view2, return_tensors="pt")

        pixel_values1 = inputs1["pixel_values"].squeeze(0)
        pixel_values2 = inputs2["pixel_values"].squeeze(0)

        return {
            "pixel_values1": pixel_values1,  # anchor
            "pixel_values2": pixel_values2,  # positive
            "labels": label
        }
