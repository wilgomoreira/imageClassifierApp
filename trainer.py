import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from model import MLPNN
import os

class TrainerW:
    def __init__(self, dataset, dataset_name, model=MLPNN, criterion=nn.CrossEntropyLoss(), learning_rate=0.0001, betas=(0.9, 0.999)):
        self.processor = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.processor)
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.input_dim, self.train_loader, self.test_loader = self._create_dataloaders()
        
        self.model =  model(self.input_dim).to(self.device).apply(self._weight_initializer)   
        self.criterion = criterion
        self.lr = learning_rate
        self.betas = betas
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)

    def _create_dataloaders(self, train_split=0.8, batch_size=32):
        self.train_split = train_split
        self.batch_size = batch_size
        # input_dim
        example_image, _ = self.dataset[0]  
        num_channels, height, width = example_image.shape
        print(f"Dataset: {self.dataset_name.upper()} with Channels: {num_channels} | Height: {height} | Width: {width}")
        input_dim = num_channels * height * width
        
        # split_dataset
        train_size = int(train_split * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_data, test_data = random_split(self.dataset, [train_size, test_size])

        # dataloaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return input_dim, train_loader, test_loader
    
    def _weight_initializer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def train(self, epochs=5):  
        self.epochs = epochs
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device).long()
                self.optimizer.zero_grad()
                logit_outputs = self.model(images)
                loss = self.criterion(logit_outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_loader):.4f}")
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing Model"):
                images, labels = images.to(self.device), labels.to(self.device).long()
                logits_outputs = self.model(images)
                predicted = torch.argmax(logits_outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.test_acc = 100 * correct / total
        print(f"Test Accuracy: {self.test_acc:.2f}%")
    
    def get_logits_labels(self):
        self.model.eval()
        train_logits, train_labels = [], []
        with torch.no_grad():
            for images, lbls in tqdm(self.train_loader, desc="Extracting Train Logits"):
                images = images.to(self.device)
                outputs = self.model(images)
                train_logits.extend(outputs.cpu().numpy())
                train_labels.extend(lbls.numpy())
        
        test_logits, test_labels = [], []
        with torch.no_grad():
            for images, lbls in tqdm(self.test_loader, desc="Extracting Test Logits"):
                images = images.to(self.device)
                outputs = self.model(images)
                test_logits.extend(outputs.cpu().numpy())
                test_labels.extend(lbls.numpy())

        self.train_logits = train_logits
        self.train_labels = train_labels
        self.test_logits = test_logits
        self.test_labels = test_labels
    
    def save_logits_labels_model(self, dir_logits_labels='logits_labels/', model_dir='model_saved/', results_dir='results/'):   
        np.save(f'{dir_logits_labels}train_logits.npy', self.train_logits)
        np.save(f'{dir_logits_labels}train_labels.npy', self.train_labels)
        np.save(f'{dir_logits_labels}test_logits.npy', self.test_logits)
        np.save(f'{dir_logits_labels}test_labels.npy', self.test_labels)
        print("logits and labels were saved successfully!")

        model_path = f'{model_dir}mpl_model_{self.dataset_name.lower()}.pth'
        torch.save(self.model.state_dict(), model_path)
        print("model was saved successfully!")

        text =  (f'MODEL HAS: \n'
                f'Number of classes: {self.model.num_classes} \n' 
                f'---------------------------------------------------\n' 
                f'MODEL WAS TRAINDED WITH:\n' 
                f'Device processor: {self.processor.upper()}\n'
                f'Dataset: {self.dataset_name.upper()}\n'
                f'Train split: {self.train_split*100}%\n'
                f'Batch size: {self.batch_size}\n'
                f'Epochs: {self.epochs}\n'
                f'Learning Rate: {self.lr}\n'
                f'Betas: {self.betas}\n'
                f'TEST ACC: {self.test_acc:.2f}%')

        with open(os.path.join(results_dir, 'train_infos.txt'), 'w') as f:
            f.write(text)