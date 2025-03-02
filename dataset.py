import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

class BinaryDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        match self.dataset_name:
            case 'MNIST':
                return self._mnist_binary()
            case 'FashionMNIST':
                return self._fashion_mnist_binary()
            case 'CIFAR10':
                return self._cifar10_binary()
            case _:
                raise ValueError(f"Dataset '{self.dataset_name}' unknown.")

    def _mnist_binary(self, root='./data'):
        dataset = datasets.MNIST(root=root, train=True, transform=self.transform, download=True)
        return [(img, 1 if label == 1 else 0) for img, label in dataset if label in [0, 1]]

    def _fashion_mnist_binary(self, root='./data'):
        dataset = datasets.FashionMNIST(root=root, train=True, transform=self.transform, download=True)
        return [(img, 1 if label == 7 else 0) for img, label in dataset if label in [0, 7]]

    def _cifar10_binary(self, root='./data'):
        dataset = datasets.CIFAR10(root=root, train=True, transform=self.transform, download=True)
        return [(img, 1 if label == 5 else 0) for img, label in dataset if label in [3, 5]]

class DataLoaderHandlerPytorch:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.binary_dataset = BinaryDataset(dataset_name).dataset
        self.input_dim = self._get_input_dim()
        self.train_data, self.test_data = self._split_dataset()
        self.train_loader, self.test_loader = self._create_dataloaders()

    def _get_input_dim(self):
        example_image, _ = self.binary_dataset[0]  
        num_channels, height, width = example_image.shape
        print(f"Dataset: {self.dataset_name} | Channels: {num_channels} | Height: {height} | Width: {width}")
        return num_channels * height * width

    def _split_dataset(self, train_split=0.8):
        train_size = int(train_split * len(self.binary_dataset))
        test_size = len(self.binary_dataset) - train_size
        return random_split(self.binary_dataset, [train_size, test_size])

    def _create_dataloaders(self, batch_size=32):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
