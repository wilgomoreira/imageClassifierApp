import torch
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from dataset import DataLoaderHandler  
from main_train_test_model import MLPNN, Config  

DATASET_NAME = Config.DATASET_NAME
SAVED_MODEL = Config.MODEL_FILE

class ImageClassifierApp:
    def __init__(self, root, dataset_name):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("400x500")
        self.root.configure(bg="#2C3E50")
        
        self.model = self.load_model()
        self.data_handler = DataLoaderHandler(dataset_name)
        
        self.frame = tk.Frame(root, bg="#34495E", padx=20, pady=20)
        self.frame.pack(pady=20)
        
        self.image_label = tk.Label(self.frame, bg="#34495E")
        self.image_label.pack()
        
        self.result_label = tk.Label(self.frame, text="Load an image to classify", font=("Arial", 14), fg="#ECF0F1", bg="#34495E")
        self.result_label.pack(pady=10)
        
        self.btn_load_sample = tk.Button(self.frame, text="Load Sample Image", command=self.load_sample_image, bg="#E74C3C", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.btn_load_sample.pack(pady=5)
        
        self.btn_open = tk.Button(self.frame, text="Open Image", command=self.open_image, bg="#3498DB", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.btn_open.pack(pady=5)
        
        self.btn_exit = tk.Button(self.frame, text="Exit", command=root.quit, bg="#2ECC71", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.btn_exit.pack(pady=5)
    
    def load_model(self):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLPNN().to(DEVICE)
        model.load_state_dict(torch.load(SAVED_MODEL, map_location=DEVICE))
        model.eval()
        return model
    
    def preprocess_image(self, image):
        image = image.convert('L').resize((28, 28))
        image_array = np.array(image) / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32).view(1, 1, 28, 28)
        return image_tensor
    
    def predict_class(self, image_tensor):
        with torch.no_grad():
            logits = self.model(image_tensor.view(1, -1))
            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability > 0.5 else 0
        return prediction, probability
    
    def load_sample_image(self):
        sample_loader = iter(self.data_handler.test_loader)
        image, label = next(sample_loader)
        image = image[0].squeeze().numpy()
        label = label[0].item()
        
        image_pil = Image.fromarray((image * 255).astype(np.uint8)).convert('L')
        self.display_image(image_pil)
        
        prediction, probability = self.predict_class(self.preprocess_image(image_pil))
        self.result_label.config(text=f"Predicted: {prediction} ({probability:.2f})\nActual: {label}")
    
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        image = Image.open(file_path).convert('L').resize((28, 28))
        self.display_image(image)
        
        prediction, probability = self.predict_class(self.preprocess_image(image))
        self.result_label.config(text=f"Predicted: {prediction} ({probability:.2f})")
    
    def display_image(self, image):
        img_display = ImageTk.PhotoImage(image.resize((150, 150)))
        self.image_label.config(image=img_display)
        self.image_label.image = img_display

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root, DATASET_NAME)
    root.mainloop()
