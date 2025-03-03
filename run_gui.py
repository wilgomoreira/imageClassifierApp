import torch
import gradio as gr
from model import MLPNN
import torchvision.transforms as transforms

class ImageClassificationApp:
    title: str
    description: str
    classes: tuple
    model: torch
    interface: gr

    def __init__(self, classes=('FIRE', 'NON FIRE')):
        self.title = f'{classes[0]} or {classes[1]} Detection'
        self.description = f'Upload an image to classify it as {classes[0]} or {classes[1]}.'
        self.classes = classes

        self.model = self._load_model()
        self.interface = self._create_interface()
        self.interface.launch()    # launch(true) for public link

    def _load_model(self, n_channels=3, resize_img=(256, 256), model_path='model_saved/mpl_model.pth'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_dim = n_channels * resize_img[0] * resize_img[1]
        model = MLPNN(input_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        return model

    def _transform(self, resize_img=(256, 256)):
        return  transforms.Compose([
                    transforms.Resize(resize_img),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])

    def _create_interface(self):
        return gr.Interface(
            fn=self._classify_image,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(),
            title=self.title,
            description=self.description,
            allow_flagging="never"
        )

    def _classify_image(self, image, threshold=0.5):
        transform = self._transform() 
        image = transform(image).unsqueeze(0)  

        with torch.no_grad():
            output = torch.sigmoid(self.model(image)).item()
            predicted_class = self.classes[1] if output >= threshold else self.classes[0]
        return predicted_class

if __name__ == "__main__":
    ImageClassificationApp()
    