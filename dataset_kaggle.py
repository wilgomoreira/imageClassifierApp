import os
import kaggle
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

DATASET_PATH_DICT = {'FIRE': 'phylake1337/fire-dataset',
                     'CATS_AND_DOGS': 'shaunthesheep/microsoft-catsvsdogs-dataset'}

class CustomImageDataset(Dataset):
    root_dir: str
    transform: transforms
    image_paths: list
    labels: list
    valid_extensions: list

    def __init__(self, root_dir, resize_img=(256, 256)):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize(resize_img),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_paths = []
        self.labels = []
        self.valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}  

        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    # Check if the file has a valid image extension
                    if any(img_file.lower().endswith(ext) for ext in self.valid_extensions):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')  # Convert to RGB to avoid grayscale issues
        except (UnidentifiedImageError, OSError):
            print(f"Error loading image: {img_path} (Skipping)")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # Get the next valid image
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class KaggleDataset:
    dataset_name: str
    extract_path: str
    dataset_path: str
    kaggle_dataset_path: str
    dataset: Dataset

    def __init__(self, dataset_name, root='./data_kaggle'):
        self.dataset_name = dataset_name
        self.extract_path = os.path.join(root, dataset_name)

        self.dataset_path = self._ensure_dataset_availability_and_download(dataset_name)
        self.dataset = CustomImageDataset(self.dataset_path)
    
    def _ensure_dataset_availability_and_download(self, dataset_name):
        dataset_path = self._get_extracted_dataset_path()
        
        if dataset_path is None:
            self.kaggle_dataset_path = DATASET_PATH_DICT[dataset_name]
            os.makedirs(self.extract_path, exist_ok=True)
            
            kaggle.api.dataset_download_files(self.kaggle_dataset_path, path=self.extract_path, unzip=True)
            print(f'Dataset {dataset_name} downloaded and extracted in {self.extract_path}')
            
            dataset_path = self._get_extracted_dataset_path()
        
        return dataset_path

    def _get_extracted_dataset_path(self):
        if not os.path.exists(self.extract_path) or not os.listdir(self.extract_path):
            return None
        # Dynamically identifies the name of the extracted dataset folder
        subdirs = [d for d in os.listdir(self.extract_path) if os.path.isdir(os.path.join(self.extract_path, d))]

        if any(os.path.isdir(os.path.join(self.extract_path, d)) for d in subdirs):
            dataset_subdir = os.path.join(self.extract_path, subdirs[0]) 
            return dataset_subdir 
        print(f"Dataset found it in: {self.extract_path}")
   

    
