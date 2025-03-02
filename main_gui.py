import torch
import gradio as gr
import numpy as np
from main_train_test_model import MLPNN

#OPTIONS OF GUI: gradio, streamlit, customtkinter, kivy, pyqt5, PySimpleGUI

class ImageClassificationApp:
    def __init__(self, model_path='model_saved/mpl_model.pth'):
        self.title = 'Fire Detection'
        self.description = 'Upload an image to classify it as FIRE or NON FIRE.'
        self.class_labels = {0: "FIRE", 1: "NON FIRE"} 

        self.model = self.load_model(model_path)
        self.interface = self.create_interface()
        self.interface.launch()

    def load_model(self, model_path, n_channels=3, resize_img=(256, 256)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_dim = n_channels * resize_img[0] * resize_img[1]
        model = MLPNN(input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        return model
    
    def create_interface(self):
        return gr.Interface(
            fn=self.classify_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(),
            title=self.title,
            description=self.description,
            allow_flagging="never"
        )

    def classify_image(self, image, resize_img=(256, 256), threshold=0.5):
        image = image.convert("RGB")
        image = image.resize((resize_img[0],resize_img[1]))  
        image_array = np.array(image) / 255.0
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output).squeeze().item()
            predicted_class = self.class_labels[int(prediction > threshold)]
        
        return predicted_class

if __name__ == "__main__":
    ImageClassificationApp()
    