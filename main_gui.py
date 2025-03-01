import torch
import gradio as gr
import numpy as np
from main_train_test_model import MLPNN, Config  

#OPTIONS OF GUI: gradio, streamlit, customtkinter, kivy, pyqt5, PySimpleGUI
RESIZE_IMG = (256, 256)   # resize the image for this value
N_CHANNELS = 3
CLASS_LABELS = {0: "FIRE", 1: "NON FIRE"} 
TITLE = 'Fire Detection'
DESCRIPTION = "Upload an image to classify it as FIRE or NON FIRE."

class ImageClassificationApp:
    def __init__(self, title, description, model_path, class_labels):
        """Initialize the model and set up the interface."""
        self.model = self.load_model(model_path)
        self.class_labels = class_labels
        self.interface = self.create_interface(title, description)
        self.interface.launch()

    def load_model(self, model_path):
        """Load the trained model."""
        input_dim = N_CHANNELS * RESIZE_IMG[0] * RESIZE_IMG[1]
        model = MLPNN(input_dim, Config.N_NEURONS, Config.N_CLASSES)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(Config.DEVICE)))
        model.eval()
        return model
    
    def create_interface(self, title, description):
        """Create the Gradio interface."""
        return gr.Interface(
            fn=self.classify_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(),
            title=title,
            description=description,
            allow_flagging="never"
        )

    def classify_image(self, image):
        """Process the image and return the predicted class."""
        image = image.convert("RGB")
        image = image.resize((RESIZE_IMG[0],RESIZE_IMG[1]))  
        image_array = np.array(image) / 255.0
        image_tensor = torch.tensor(image_array).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output).squeeze().item()
            predicted_class = self.class_labels[round(prediction)]
        
        return predicted_class

if __name__ == "__main__":
    ImageClassificationApp(TITLE, DESCRIPTION, Config.MODEL_FILE, CLASS_LABELS)
    