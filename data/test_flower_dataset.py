import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FlowerDataset(Dataset):
    def __init__(self, root_dir, image_size=32, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_size (int): Size to resize images to.
            mode (str): 'train' or 'val' to select training or validation set.
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.image_size = image_size
        self.classes = sorted(os.listdir(self.root_dir))
        self.image_paths = []
        
        # Collect all image paths
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # For diffusion models, we don't need labels
        return image, 0  # Return 0 as dummy label