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
        self._load_data()
        self._divide_classes_from_model()
        self._run_analysis()
         
    
    def _load_data(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.PLOTS_DIR, exist_ok=True)

        dir_logits_labels = Config.DIR_LOGITS_LABELS
        self.train_logits = np.load(f'{dir_logits_labels}train_logits.npy')
        self.train_labels = np.load(f'{dir_logits_labels}train_labels.npy')
        
        self.test_logits = np.load(f'{dir_logits_labels}test_logits.npy')
        self.test_labels = np.load(f'{dir_logits_labels}test_labels.npy')
    
    def _divide_classes_from_model(self):
        self.classes_train_logits = self._separate_logits_by_class(self.train_logits, self.train_labels)
        self.classes_test_logits = self._separate_logits_by_class(self.test_logits, self.test_labels)

    def _separate_logits_by_class(self, logits, labels):
        logits = np.array(logits).squeeze()
        labels = np.array(labels)
        
        unique_classes = np.unique(labels)
        class_logits = {cls: logits[labels == cls] for cls in unique_classes}
        return class_logits

    def _run_analysis(self):
        self._generate_histograms(self.classes_test_logits, 'logit', '1')
        likelihoods = self._convert_logits_to_likelihoods(self.classes_test_logits)
        
        self._generate_histograms(likelihoods, 'likelihood', '2')
        kde = BinaryKDE(self.classes_train_logits, self.test_logits)
        self._compute_metrics(self.test_logits, kde.posterior_probs, self.test_labels, 'METRICS IN TEST TIME')
    
    def _generate_histograms(self, data_dict, name, identifier):
        plt.figure(figsize=(8, 6))
        for cls, data in data_dict.items():
            plt.hist(data.flatten(), bins=self.BIN_HIST, histtype='step', linewidth=1.5, label=f'Class {cls}', density=True)
        
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {name}')
        plt.savefig(f"{self.PLOTS_DIR}/{identifier}_{name}_histogram.pdf")
        plt.close()
    
    def _compute_metrics(self, logits, enhanced_probs, labels, name):
        true_labels = labels.flatten()
        baseline_probs = self._logits_to_likelihoods(logits.flatten())
        metrics_baseline = self._calculate_metrics(true_labels, baseline_probs)
        
        metrics_enhanced = self._calculate_metrics(true_labels, enhanced_probs.flatten())
        result_text = self._format_metrics(name, metrics_baseline, metrics_enhanced)
        self._save_results(result_text, f'{name.lower()}.txt')
    
    def _calculate_metrics(self, true_labels, probs):
        predicted_labels = (probs > self.THRESHOLD).astype(int)
        return {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'f1_score': f1_score(true_labels, predicted_labels),
            'avg_precision': average_precision_score(true_labels, probs),
            'ece': ECE(bins=self.BIN_ECE).measure(probs, true_labels)
        }
    
    def _format_metrics(self, name, baseline, enhanced):
        return (f'{name}\n'
                f'BASELINE:\n'
                f'ACC: {self._percent_format(baseline["accuracy"])}% | '
                f'F1 SCORE: {self._percent_format(baseline["f1_score"])}% | '
                f'AVG PREC: {self._percent_format(baseline["avg_precision"])}% | '
                f'ECE: {self._percent_format(baseline["ece"])}%\n'
                f'KDE:\n'
                f'ACC: {self._percent_format(enhanced["accuracy"])}% | '
                f'F1 SCORE: {self._percent_format(enhanced["f1_score"])}% | '
                f'AVG PREC: {self._percent_format(enhanced["avg_precision"])}% | '
                f'ECE: {self._percent_format(enhanced["ece"])}%\n')
    
    def _logits_to_likelihoods(self, logits):
        return torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    
    def _convert_logits_to_likelihoods(self, class_dict):
        return {cls: self._logits_to_likelihoods(logits) for cls, logits in class_dict.items()}
    
    def _percent_format(self, number):
        return round(100 * number, self.ROUND)
    
    def _save_results(self, text, filename):
        with open(os.path.join(self.OUTPUT_DIR, filename), 'w') as f:
            f.write(text)

if __name__ == "__main__":
    PostProcessing()
    print("FINISH!!")
