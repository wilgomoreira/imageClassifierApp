import argparse
from dataset_torch import TorchDataset
from dataset_kaggle import KaggleDataset
from trainer import TrainerW


# ORING DATA: TORCH, KAGGLE
# TORCH DATA: MNIST, FashionMNIST, CIFAR10
# KAGGLE DATA: FIRE

class ArgumentParserHandler:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Train a neural network on a selected dataset.")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--origin_data", type=str, choices=['TORCH', 'KAGGLE'], default='KAGGLE',
                                 help="Choose the dataset source: TORCH or KAGGLE.")
        self.parser.add_argument("--torch_data", type=str, choices=['MNIST', 'FashionMNIST', 'CIFAR10'], default='CIFAR10',
                                 help="Select a dataset from TORCH (only used if dataset_origin is TORCH).")
        self.parser.add_argument("--kaggle_data", type=str, choices=['FIRE', 'CATS_AND_DOGS'], default='FIRE',
                                 help="Select a dataset from KAGGLE (only used if dataset_origin is KAGGLE).")
        self.parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
        self.parser.add_argument("--lr", type=int, default=0.0001, help="learning rate of training.")

    def parse_arguments(self):
        return self.parser.parse_args()

def select_dataset(origin_data='KAGGLE', torch_data='CIFAR10', kaggle_data='FIRE'):
    if origin_data == 'TORCH':
        torch_dataset = TorchDataset(torch_data)
        dataset = torch_dataset.dataset
        dataset_name = f'{torch_dataset.dataset_name}-torch'
    else:
        kaggle_dataset = KaggleDataset(kaggle_data)
        dataset = kaggle_dataset.dataset
        dataset_name = f'{kaggle_dataset.dataset_name}-kaggle'
    
    return dataset, dataset_name

def main():
    args_handler = ArgumentParserHandler()
    args = args_handler.parse_arguments()

    dataset, dataset_name = select_dataset(args.origin_data, args.torch_data, args.kaggle_data)
    trainer = TrainerW(dataset, dataset_name) 
    trainer.train(args.epochs)
    trainer.test()
        
    trainer.get_logits_labels()
    trainer.save_logits_labels_model()
    print("FINISHED!!")

if __name__ == "__main__":
    main()
    
    
    
