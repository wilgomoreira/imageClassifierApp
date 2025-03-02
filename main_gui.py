import torch
import gradio as gr
import numpy as np
from main_train_test_model import MLPNN
import torchvision.transforms as transforms

#OPTIONS OF GUI: gradio, streamlit, customtkinter, kivy, pyqt5, PySimpleGUI

class ImageClassificationApp:
    def __init__(self):
        self.title = 'Fire Detection'
        self.description = 'Upload an image to classify it as FIRE or NON FIRE.'
        self.classes = ("FIRE", "NON FIRE")

        self.model = self.load_model()
        self.interface = self.create_interface()
        self.interface.launch()    # launch(true) for public link

    def load_model(self, n_channels=3, resize_img=(256, 256), model_path='model_saved/mpl_model.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_dim = n_channels * resize_img[0] * resize_img[1]
        model = MLPNN(input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        return model

    def transform(self, resize_img=(256, 256)):
        return  transforms.Compose([
                    transforms.Resize(resize_img),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])

    def create_interface(self):
        return gr.Interface(
            fn=self.classify_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(),
            title=self.title,
            description=self.description,
            allow_flagging="never"
        )

    def classify_image(self, image, threshold=0.5):
        transform = self.transform() 
        image = transform(image).unsqueeze(0)  

        with torch.no_grad():
            output = torch.sigmoid(self.model(image)).item()
            predicted_class = self.classes[1] if output >= threshold else self.classes[0]
        return predicted_class

if __name__ == "__main__":
    ImageClassificationApp()
    