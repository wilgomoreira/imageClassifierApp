# Neural Network Training with Torch and Kaggle Datasets

## Overview
This project provides a framework to train a **binary classifier** neural network using different datasets from Torch and Kaggle. The training pipeline allows for dataset selection, model training, evaluation, and saving the model along with logits and labels.
# Neural Network Training with Torch and Kaggle Datasets

## Overview
This project provides a framework to train a **binary classifier** neural network using different datasets from Torch and Kaggle. The training pipeline allows for dataset selection, model training, evaluation, and saving the model along with logits and labels.

## Features
- Supports multiple datasets:
  - **Torch datasets**: MNIST, FashionMNIST, CIFAR10
  - **Kaggle datasets**: FIRE
- Implements a **Multi-Layer Perceptron (MLP)** neural network for **binary classification**.
- Provides an **argument parser** for easy configuration.
- Uses **Adam optimizer** for training.
- Saves the **model, logits, and labels** after training.
- Supports **GPU acceleration** if available.

## Requirements
Ensure you have Python installed along with the following dependencies:

```bash
pip install -r requirements.txt
```

### Required Python Packages:
- `torch`
- `torchvision`
- `numpy`
- `tqdm`

## Usage

### Running the Training Script
Execute the following command to train the model:

```bash
python train.py --origin_data KAGGLE --kaggle_data FIRE --epochs 10 --lr 0.001 --threshold 0.5
```

### Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--origin_data` | `str` | KAGGLE | Data source: `TORCH` or `KAGGLE`. |
| `--torch_data` | `str` | CIFAR10 | Dataset from Torch (MNIST, FashionMNIST, CIFAR10). |
| `--kaggle_data` | `str` | FIRE | Dataset from Kaggle (FIRE). |
| `--epochs` | `int` | 5 | Number of training epochs. |
| `--lr` | `float` | 0.0001 | Learning rate. |
| `--threshold` | `float` | 0.5 | Classification threshold for testing. |

### Training and Testing
- The project as a whole **trains the model, tests it, and saves the logits, labels, and trained model** for future use.
- The model will be trained on the selected dataset for the specified number of epochs.
- This neural network is a **binary classifier**, meaning it is designed to differentiate between two classes.
- After training, the model will be evaluated on the test dataset.
- The logits and labels will be saved for further analysis.

### Additional Scripts

#### Post-Processing
There is an additional script for post-processing the logits and labels:
```bash
python post_processing.py
```
This script loads the **train and test logits and labels** and allows for further analysis, such as:
- Generating **histograms**.
- Applying **new approaches**.
- Calculating **additional metrics**.

#### Graphical User Interface (GUI) for Model Testing
Another script provides a graphical interface to test the trained model:
```bash
python run_gui.py
```
This interface allows users to **load images** and see predictions made by the trained model.

### Output
- Training progress with **loss tracking**.
- Test accuracy after training.
- Logits and labels saved in `logits_labels/`
- Trained model saved in `model_saved/mpl_model.pth`

## Example Output
```
Epoch 1/5, Loss: 0.4321
Epoch 2/5, Loss: 0.3289
...
Test Accuracy: 89.45%
logits and labels were saved successfully!
model was saved successfully!
FINISH!!
```

## Model Saving
- Logits and labels are saved as `.npy` files.
- The trained model is saved as a `.pth` file and can be reloaded for inference.

## License
This project is licensed under the MIT License.


