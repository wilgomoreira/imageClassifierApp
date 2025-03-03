import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DataLoaderHandlerPytorch
from dataset_kaggle import DataLoaderHandlerKaggle
import numpy as np
from tqdm import tqdm
from model import MLPNN

# ORING DATA: PYTORCH, KAGGLE
# PYTORCH DATA: MNIST, FashionMNIST, CIFAR10
# KAGGLE DATA: FIRE

class Trainer:
    def __init__(self, oring_data='KAGGLE', pytorch_data='CIFAR10', kaggle_data='FIRE',
                 model=MLPNN, learning_rate=0.0001, betas=(0.9, 0.999)):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim, self.train_loader, self.test_loader = self._select_dataset(oring_data, pytorch_data, kaggle_data)  
        self.model =  model(self.input_dim).to(self.device).apply(self._weight_initializer)   
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas)

    def _select_dataset(self, oring_data, pytorch_data, kaggle_data): 
        if oring_data == 'PYTORCH':
            dataloader = DataLoaderHandlerPytorch(pytorch_data)
        else:
            dataloader = DataLoaderHandlerKaggle(kaggle_data)

        return dataloader.input_dim, dataloader.train_loader, dataloader.test_loader
    
    def _weight_initializer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def train(self, n_epochs=5):  
        for epoch in range(n_epochs):
            self.model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device).float().view(-1, 1)
                self.optimizer.zero_grad()
                logit_outputs = self.model(images)
                loss = self.criterion(logit_outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
            
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(self.train_loader):.4f}")
    
    def test(self, threshold=0.5):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits_outputs = self.model(images)
                like_outputs = torch.sigmoid(logits_outputs)
                predicted = (like_outputs > threshold).float()
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
    
    def get_logits_labels(self):
        self.model.eval()

        train_logits, train_labels = [], []
        with torch.no_grad():
            for images, lbls in self.train_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                train_logits.extend(outputs.cpu().numpy())
                train_labels.extend(lbls.numpy())
        
        test_logits, test_labels = [], []
        with torch.no_grad():
            for images, lbls in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                test_logits.extend(outputs.cpu().numpy())
                test_labels.extend(lbls.numpy())

        self.train_logits = train_logits
        self.train_labels = train_labels
        self.test_logits = test_logits
        self.test_labels = test_labels
    
    def save_logits_labels_model(self, dir_logits_labels='logits_labels/', model_path='model_saved/mpl_model.pth'):   
        np.save(f'{dir_logits_labels}train_logits.npy', self.train_logits)
        np.save(f'{dir_logits_labels}train_labels.npy', self.train_labels)
        np.save(f'{dir_logits_labels}test_logits.npy', self.test_logits)
        np.save(f'{dir_logits_labels}test_labels.npy', self.test_labels)
        print("logits and labels were saved successfully!")

        torch.save(self.model.state_dict(), model_path)
        print("model was saved successfully!")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.test()
        
    trainer.get_logits_labels()
    trainer.save_logits_labels_model()
    print("FINISH!!")
    
    
