import tkinter as tk
from PIL import Image, ImageTk
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Carregar o dataset CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Índice inicial da imagem
current_index = 0

# Criar a janela principal
root = tk.Tk()
root.title("Visualizador CIFAR-10")

# Função para exibir a imagem atual
def show_image():
    global current_index
    img, label = dataset[current_index]
    img = np.transpose(img.numpy(), (1, 2, 0))  # Convertendo para HxWxC
    img = (img * 255).astype(np.uint8)  # Convertendo para valores de imagem

    # Convertendo para o formato PIL e exibindo na interface
    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_pil)
    canvas.create_image(150, 150, anchor=tk.CENTER, image=img_tk)
    canvas.image = img_tk  # Manter referência para evitar garbage collection
    label_widget.config(text=f"Índice: {current_index} | Classe: {dataset.classes[label]}")

# Função para avançar para a próxima imagem
def next_image():
    global current_index
    if current_index < len(dataset) - 1:
        current_index += 1
        show_image()

# Função para voltar para a imagem anterior
def prev_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image()

# Criar um canvas para exibir a imagem
canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

# Criar um label para mostrar a classe da imagem
label_widget = tk.Label(root, text="Índice: 0 | Classe: ")
label_widget.pack()

# Criar botões para navegação
btn_prev = tk.Button(root, text="Anterior", command=prev_image)
btn_prev.pack(side=tk.LEFT, padx=20, pady=20)

btn_next = tk.Button(root, text="Próximo", command=next_image)
btn_next.pack(side=tk.RIGHT, padx=20, pady=20)

# Mostrar a primeira imagem
show_image()

# Iniciar a interface gráfica
root.mainloop()
