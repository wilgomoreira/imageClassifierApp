import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DataLoaderHandler
from dataset_kaggle import DataLoaderHandler as DataLoaderHandlerKaggle
from dataset_kaggle import Config as ConfigDataKaggle
import numpy as np

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DIR_LOGITS_LABELS = 'logits_labels/'
    MODEL_FILE = 'model_saved/mpl_model.pth'
    N_NEURONS = 128
    N_CLASSES = 1  # BINARY: 1, OTHERS: NUMBER OF CLASSES
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    DATASET_ORIGIN = 'KAGGLE' # PYTORCH, KAGGLE
    DATASET_NAME_PYTORCH = 'CIFAR10'  # BINARY DATASET: MNIST, FashionMNIST, CIFAR10
    DATASET_NAME_KAGGLE = 'FIRE'   # FIRE

class ChooseDataset:
    def __init__(self, dataset_origin):
        if dataset_origin == 'PYTORCH':
            dataset_name = Config.DATASET_NAME_PYTORCH
            dataloader = DataLoaderHandler(dataset_name)
        else:
            dataset_name = Config.DATASET_NAME_KAGGLE
            dataset_path = ConfigDataKaggle.DATASET_PATH_KAGGLE.get(dataset_name, {})
            dataloader = DataLoaderHandlerKaggle(dataset_name, dataset_path)

        self.input_dim = dataloader.input_dim
        self.train_loader = dataloader.train_loader
        self.test_loader = dataloader.test_loader

class MLPNN(nn.Module):
    def __init__(self, input_dim, num_neurons, num_classes):
        super(MLPNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_classes)  
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader):
        self.model = model.to(Config.DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train(self):  
        for epoch in range(Config.NUM_EPOCHS):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE).float().view(-1, 1)
                self.optimizer.zero_grad()
                logit_outputs = self.model(images)
                loss = self.criterion(logit_outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {running_loss/len(self.train_loader):.4f}")
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                logits_outputs = self.model(images)
                like_outputs = torch.sigmoid(logits_outputs)
                predicted = (like_outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def get_logits_labels(self, data_loader):
        self.model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for images, lbls in data_loader:
                images = images.to(Config.DEVICE)
                outputs = self.model(images)
                logits.extend(outputs.cpu().numpy())
                labels.extend(lbls.numpy())
        return logits, labels
    
    def save_logits_labels(self, train_logits, train_labels, test_logits, test_labels):
        np.save(f'{Config.DIR_LOGITS_LABELS}train_logits.npy', train_logits)
        np.save(f'{Config.DIR_LOGITS_LABELS}train_labels.npy', train_labels)
        np.save(f'{Config.DIR_LOGITS_LABELS}test_logits.npy', test_logits)
        np.save(f'{Config.DIR_LOGITS_LABELS}test_labels.npy', test_labels)
        print("logits and labels were saved successfully!")

class RunTraining:
    def __init__(self):
        # Load dataset
        dataset = ChooseDataset(Config.DATASET_ORIGIN)
        input_dim, train_loader, test_loader = dataset.input_dim, dataset.train_loader, dataset.test_loader
        
        # Initialize model, loss, optimizer
        model = MLPNN(input_dim, Config.N_NEURONS, Config.N_CLASSES)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Train and evaluate
        trainer = Trainer(model, criterion, optimizer, train_loader, test_loader)
        trainer.train()
        trainer.test()

        # Save the trained model
        trainer.save_model(Config.MODEL_FILE)
        
        # Generate logits and labels, then save them
        train_logits, train_labels = trainer.get_logits_labels(train_loader)
        test_logits, test_labels = trainer.get_logits_labels(test_loader)
        trainer.save_logits_labels(train_logits, train_labels, test_logits, test_labels)    

if __name__ == "__main__":
    RunTraining()
    print("FINISH!!")
    
    
