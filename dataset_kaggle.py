import os
import kaggle
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image

def download_and_extract_kaggle_dataset(dataset_name, kaggle_dataset_path, extract_path="./data"):
    os.makedirs(extract_path, exist_ok=True)
    kaggle.api.dataset_download_files(kaggle_dataset_path, path=extract_path, unzip=True)
    print(f"Dataset {dataset_name} baixado e extraído em {extract_path}")

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DataLoaderHandler:
    def __init__(self, dataset_name, kaggle_dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join("./data", dataset_name)
        
        if not os.path.exists(self.dataset_path):
            download_and_extract_kaggle_dataset(dataset_name, kaggle_dataset_path, self.dataset_path)
        
        self.dataset = CustomImageDataset(self.dataset_path)
        self.input_dim = self._get_input_dim()
        self.train_data, self.test_data = self._split_dataset()
        self.train_loader, self.test_loader = self._create_dataloaders()

    def _get_input_dim(self):
        example_image, _ = self.dataset[0]
        num_channels, height, width = example_image.shape
        print(f"Dataset: {self.dataset_name} | Channels: {num_channels} | Height: {height} | Width: {width}")
        return num_channels * height * width

    def _split_dataset(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        return random_split(self.dataset, [train_size, test_size])

    def _create_dataloaders(self):
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=32, shuffle=False)
        return train_loader, test_loader
