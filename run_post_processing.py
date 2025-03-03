import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from kde_inter_prob import BinaryKDE

class PostProcessing:
    def __init__(self):
        self._load_data()
        self._divide_classes_from_model()
         
    def _load_data(self, out_dir='results', dir_logits_labels='logits_labels/'):
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

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

    def run_analysis(self):
        # Baseline of logits and likelihoods
        self._generate_histograms(self.classes_test_logits, 'logit', '1')
        likelihoods = {cls: self._logits_to_likelihoods(logits) for cls, logits in self.classes_test_logits.items()}
        self._generate_histograms(likelihoods, 'likelihood', '2')
        
        # Using KDE approach
        kde = BinaryKDE(self.classes_train_logits, self.test_logits)
        posterior_probs = kde.compute_posterior_prob()
        
        # Evaluate baseline with KDE
        self._compute_metrics(self.test_logits, posterior_probs, self.test_labels, 'METRICS IN TEST TIME')
    
    def _generate_histograms(self, data_dict, name, identifier, plot_dir='plots_and_histograms', bin=20):
        plt.figure(figsize=(8, 6))
        for cls, data in data_dict.items():
            plt.hist(data.flatten(), bins=bin, histtype='step', linewidth=1.5, label=f'Class {cls}', density=True)
        
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {name}')
        plt.savefig(f"{plot_dir}/{identifier}_{name}_histogram.pdf")
        plt.close()
    
    def _compute_metrics(self, logits, enhanced_probs, labels, name):
        true_labels = labels.flatten()
        # Baseline
        baseline_probs = self._logits_to_likelihoods(logits.flatten())
        metrics_baseline = self._calculate_metrics(true_labels, baseline_probs)
        
        # Other approach
        metrics_enhanced = self._calculate_metrics(true_labels, enhanced_probs.flatten())
        result_text = self._format_metrics(name, metrics_baseline, metrics_enhanced)

        self._save_results(result_text, f'{name.lower()}.txt')
    
    def _calculate_metrics(self, true_labels, probs, threshold=0.5, bin=10):
        predicted_labels = (probs > threshold).astype(int)
        return {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'f1_score': f1_score(true_labels, predicted_labels),
            'avg_precision': average_precision_score(true_labels, probs),
            'ece': ECE(bins=bin).measure(probs, true_labels)
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
    
    def _percent_format(self, number, round_number=2):
        return round(100 * number, round_number)
    
    def _save_results(self, text, filename, output_dir='results'):
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(text)

if __name__ == "__main__":
    post_process = PostProcessing()
    post_process.run_analysis()
    print("FINISH!!")
