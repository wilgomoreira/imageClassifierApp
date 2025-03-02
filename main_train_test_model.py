import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DataLoaderHandlerPytorch
from dataset_kaggle import DataLoaderHandlerKaggle
import numpy as np
from tqdm import tqdm
   
class ChooseDataset:
    def __init__(self, user_dataset):
        if user_dataset['origin'] == 'PYTORCH':
            dataloader = DataLoaderHandlerPytorch(user_dataset['pytorch'])
        else:
            dataloader = DataLoaderHandlerKaggle(user_dataset['kaggle'])

        self.input_dim = dataloader.input_dim
        self.train_loader = dataloader.train_loader
        self.test_loader = dataloader.test_loader

class MLPNN(nn.Module):
    def __init__(self, input_dim, num_neurons=128, num_classes=1):
        super(MLPNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, num_classes)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train(self, n_epochs=15):  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(n_epochs):
            self.model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                self.optimizer.zero_grad()
                logit_outputs = self.model(images)
                loss = self.criterion(logit_outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
            
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(self.train_loader):.4f}")
    
    def test(self, threshold=0.5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                logits_outputs = self.model(images)
                like_outputs = torch.sigmoid(logits_outputs)
                predicted = (like_outputs > threshold).float()
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    def save_model(self, path='model_saved/mpl_model.pth'):
        torch.save(self.model.state_dict(), path)
        print("model was saved successfully!")

    def get_logits_labels(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for images, lbls in data_loader:
                images = images.to(device)
                outputs = self.model(images)
                logits.extend(outputs.cpu().numpy())
                labels.extend(lbls.numpy())
        return logits, labels
    
    def save_logits_labels(self, train_logits, train_labels, test_logits, test_labels, dir_logits_labels='logits_labels/'):
        np.save(f'{dir_logits_labels}train_logits.npy', train_logits)
        np.save(f'{dir_logits_labels}train_labels.npy', train_labels)
        np.save(f'{dir_logits_labels}test_logits.npy', test_logits)
        np.save(f'{dir_logits_labels}test_labels.npy', test_labels)
        print("logits and labels were saved successfully!")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class RunTraining:
    def __init__(self, user_dataset):
        # Load dataset
        dataset = ChooseDataset(user_dataset)
        input_dim, train_loader, test_loader = dataset.input_dim, dataset.train_loader, dataset.test_loader
        
        # Initialize model, loss, optimizer
        model = MLPNN(input_dim)
        model.apply(init_weights) 
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        # Train and evaluate
        trainer = Trainer(model, criterion, optimizer, train_loader, test_loader)
        trainer.train()
        trainer.test()

        # Save the trained model
        trainer.save_model()
        
        # Generate logits and labels, then save them
        train_logits, train_labels = trainer.get_logits_labels(train_loader)
        test_logits, test_labels = trainer.get_logits_labels(test_loader)
        trainer.save_logits_labels(train_logits, train_labels, test_logits, test_labels)    

if __name__ == "__main__":
    user_dataset = {'origin': 'KAGGLE',    # PYTORCH, KAGGLE
                    'pytorch': 'CIFAR10',  # BINARY DATASET: MNIST, FashionMNIST, CIFAR10
                    'kaggle': 'FIRE'       # FIRE
    }

    RunTraining(user_dataset)
    print("FINISH!!")
    
    
