import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from main_train_test_model import Config
from inter_prob import BinaryKDE

class PostProcessing:
    THRESHOLD = 0.5
    OUTPUT_DIR = 'results'
    PLOTS_DIR = 'plots_and_histograms'
    BIN_ECE = 10
    BIN_HIST = 30
    ROUND = 2
    
    def __init__(self):
        self.dir_logits_labels = Config.DIR_LOGITS_LABELS
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
        
        # Likelihoods - Histogram and metrics
        classes_likes = self.logits_to_like_in_all_classes(self.classes_logits)
        self.histogram(classes_likes, 'likelihood', '2')

        #likelihoods by KDE
        kde = BinaryKDE(self.classes_logits, self.test_labels)
        self.compute_metrics(self.test_logits, kde.posterior_probs, self.test_labels, 'METRICS IN TEST TIME')

    
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
    
    def compute_metrics(self, baseline_logits, enhanced_like, test_labels, name):
        baseline_logits_flatt = baseline_logits.flatten()
        baseline_like_flatt = self.logits_to_likelihoods(baseline_logits_flatt)
        true_labels_flatt = test_labels.flatten()
        baseline_labels_flatt = [1 if prob > self.THRESHOLD else 0 for prob in baseline_like_flatt]
        
        #baseline
        base_acc = accuracy_score(true_labels_flatt, baseline_labels_flatt)
        base_f1 = f1_score(true_labels_flatt, baseline_labels_flatt)
        base_ave_prec = average_precision_score(true_labels_flatt, baseline_like_flatt)
        base_ece = ECE(bins=self.BIN_ECE)
        base_ece_score = base_ece.measure(baseline_like_flatt, true_labels_flatt)

        #enhanced probs
        enhanced_like_flatt = enhanced_like.flatten()
        enhanced_labels_flatt = [1 if prob > self.THRESHOLD else 0 for prob in enhanced_like_flatt]
        enhanced_acc = accuracy_score(true_labels_flatt, enhanced_labels_flatt)
        enhanced_f1 = f1_score(true_labels_flatt, enhanced_labels_flatt)
        enhanced_ave_prec = average_precision_score(true_labels_flatt, enhanced_like_flatt)
        enhanced_ece = ECE(bins=self.BIN_ECE)
        enhanced_ece = enhanced_ece.measure(enhanced_like_flatt, true_labels_flatt)

        result_text = (f'{name}\n'
                       f'BASELINE: \n'
                       f'ACC: {self.perc_format(base_acc)}% | F1 SCORE: {self.perc_format(base_f1)}% | '
                       f'AVER_PREC: {self.perc_format(base_ave_prec)}% | ECE: {self.perc_format(base_ece_score)}%\n'
                       f'KDE: \n'
                       f'ACC: {self.perc_format(enhanced_acc)}% | F1 SCORE: {self.perc_format(enhanced_f1)}% | '
                       f'AVER_PREC: {self.perc_format(enhanced_ave_prec)}% | ECE: {self.perc_format(enhanced_ece)}%\n')
        
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
