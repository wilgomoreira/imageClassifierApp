import torchvision.transforms as transforms
import torchvision.datasets as datasets

class TorchDataset:
    dataset_name: str
    transform: transforms
    dataset: datasets

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


