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
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # Map class names to indices
        self.image_paths = []
        self.labels = []
        
        # Collect all image paths and corresponding labels
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])  # Store class index
        
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
        label = self.labels[idx]  # Get the actual class index
        
        if self.transform:
            image = self.transform(image)
        
        return image, label  # Return both image and its class label