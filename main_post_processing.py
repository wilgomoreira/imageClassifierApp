import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from main_train_test_model import Config

class PostProcessing:
    THRESHOLD = 0.5
    DIR_LOGITS_LABELS = Config.DIR_LOGITS_LABELS
    OUTPUT_DIR = 'results'
    PLOTS_DIR = 'plots_and_histograms'
    BIN_ECE = 10
    BIN_HIST = 30
    ROUND = 2
    
    def __init__(self):
        self.dir_logits_labels = self.DIR_LOGITS_LABELS
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.PLOTS_DIR, exist_ok=True)
        self.train_logits, self.train_labels, self.test_logits, self.test_labels = self.load_logits_labels()
        self.classes_logits = self.separate_logits_by_class(self.test_logits, self.test_labels)
        
    def load_logits_labels(self):
        train_logits = np.load(f'{self.dir_logits_labels}train_logits.npy')
        train_labels = np.load(f'{self.dir_logits_labels}train_labels.npy')
        test_logits = np.load(f'{self.dir_logits_labels}test_logits.npy')
        test_labels = np.load(f'{self.dir_logits_labels}test_labels.npy')
        return train_logits, train_labels, test_logits, test_labels
    
    def separate_logits_by_class(self, logits, labels):
        logits = np.array(logits).squeeze()
        labels = np.array(labels)
        unique_classes = np.unique(labels)
        class_logits = {cls: [] for cls in unique_classes}
        
        for logit, label in zip(logits, labels):
            class_logits[label].append(logit)
        
        for cls in class_logits:
            class_logits[cls] = np.array(class_logits[cls])
        
        return class_logits
    
    def run_analysis(self):
        # Logits - Histogram and metrics
        self.histogram(self.classes_logits, 'logit', '1')
        self.compute_metrics(self.test_logits, self.test_labels, 'METRICS IN TEST TIME')
        
        # Likelihoods - Histogram and metrics
        classes_likes = self.logits_to_like_in_all_classes(self.classes_logits)
        self.histogram(classes_likes, 'likelihood', '2')
    
    def histogram(self, class_data_dict, name_of_chart, number):
        plt.figure(figsize=(8, 6))
        for cls, data in class_data_dict.items():
            data_flat = np.array(data).flatten()
            plt.hist(data_flat, bins=self.BIN_HIST, histtype='step', linewidth=1.5, label=f'Classe {cls}', density=True)
        
        plt.xlabel(name_of_chart)
        plt.ylabel('Frequence')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {name_of_chart}')
        plt.savefig(f"{self.PLOTS_DIR}/{number}_{name_of_chart}_histogram.pdf")
        plt.close()
    
    def compute_metrics(self, logits, true_labels, name):
        logits_flat = logits.flatten()
        like_flat = self.logits_to_likelihoods(logits_flat)
        true_labels_flat = true_labels.flatten()
        pred_labels = [1 if prob > self.THRESHOLD else 0 for prob in like_flat]
        
        acc = accuracy_score(true_labels_flat, pred_labels)
        f1 = f1_score(true_labels_flat, pred_labels)
        ave_prec = average_precision_score(true_labels_flat, like_flat)
        ece = ECE(bins=self.BIN_ECE)
        ece_score = ece.measure(like_flat, true_labels_flat)
        
        result_text = (f'{name}\n'
                       f'ACC: {self.perc_format(acc)}% | F1 SCORE: {self.perc_format(f1)}% | '
                       f'AVER_PREC: {self.perc_format(ave_prec)}% | ECE: {self.perc_format(ece_score)}%\n')
        
        self.save_results_to_file(result_text, f'{name.lower()}.txt')
    
    def logits_to_likelihoods(self, logits):
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        likelihoods_tensor = torch.sigmoid(logits_tensor)
        return likelihoods_tensor.numpy()
    
    def perc_format(self, number):
        return round(100 * number, self.ROUND)
    
    def logits_to_like_in_all_classes(self, class_dict):
        return {key: [self.logits_to_likelihoods(value) for value in values] for key, values in class_dict.items()}
    
    def save_results_to_file(self, result_text, filename):
        filepath = os.path.join(self.OUTPUT_DIR, filename)
        with open(filepath, 'w') as f:
            f.write(result_text)
    
if __name__ == "__main__":
    postprocessor = PostProcessing()
    postprocessor.run_analysis()
    print("FINISH!!")
