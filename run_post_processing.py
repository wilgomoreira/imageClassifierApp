import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from kde_inter_prob import sklearnKDE

class Analysis:
    output_dir: str

    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_histograms(self, data_dict, name, identifier, plot_dir='plots_and_histograms', bins=20):
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for cls, data in data_dict.items():
            plt.hist(data.flatten(), bins=bins, histtype='step', linewidth=1.5, label=f'Class {cls}', density=True)

        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {name}')
        plt.savefig(f"{plot_dir}/{identifier}_{name}_histogram.pdf")
        plt.close()

    def compute_metrics(self, logits, enhanced_probs, labels, name):
        true_labels = labels.flatten()
        
        # Baseline
        baseline_probs = self.logits_to_likelihoods(logits.flatten())
        metrics_baseline = self.calculate_metrics(true_labels, baseline_probs)
        
        # Enhanced approach
        metrics_enhanced = self.calculate_metrics(true_labels, enhanced_probs.flatten())
        self.format_and_save_results(name, metrics_baseline, metrics_enhanced)
    
    def logits_to_likelihoods(self, logits):
        return torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()

    def calculate_metrics(self, true_labels, probs, threshold=0.5, bins=10):
        predicted_labels = (probs > threshold).astype(int)
        return {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'f1_score': f1_score(true_labels, predicted_labels),
            'avg_precision': average_precision_score(true_labels, probs),
            'ece': ECE(bins=bins).measure(probs, true_labels)
        }

    def format_and_save_results(self, name, baseline, enhanced):
        text = (f'{name}\n'
                f'BASELINE:\n'
                f'ACC: {self.percent_format(baseline["accuracy"])}% | '
                f'F1 SCORE: {self.percent_format(baseline["f1_score"])}% | '
                f'AVG PREC: {self.percent_format(baseline["avg_precision"])}% | '
                f'ECE: {self.percent_format(baseline["ece"])}%\n'
                f'KDE:\n'
                f'ACC: {self.percent_format(enhanced["accuracy"])}% | '
                f'F1 SCORE: {self.percent_format(enhanced["f1_score"])}% | '
                f'AVG PREC: {self.percent_format(enhanced["avg_precision"])}% | '
                f'ECE: {self.percent_format(enhanced["ece"])}%\n')

        with open(os.path.join(self.output_dir, f'{name.lower()}.txt'), 'w') as f:
            f.write(text)
    
    def percent_format(self, number, round_number=2):
        return round(100 * number, round_number)

class PostProcessing:
    analysis: Analysis
    dir_logits_labels: str
    train_logits: np
    train_labels: np
    test_logits: np
    test_labels: np
    classes_train_logits: dict
    classes_test_logits: dict

    def __init__(self, analysis, dir_logits_labels='logits_labels/'):
        self.analysis = analysis
        self.dir_logits_labels = dir_logits_labels
        
        self._load_data()
        self._divide_classes_from_model()

    def _load_data(self):
        os.makedirs(self.analysis.output_dir, exist_ok=True)
        os.makedirs(self.dir_logits_labels, exist_ok=True)

        self.train_logits = np.load(f'{self.dir_logits_labels}train_logits.npy')
        self.train_labels = np.load(f'{self.dir_logits_labels}train_labels.npy')
        self.test_logits = np.load(f'{self.dir_logits_labels}test_logits.npy')
        self.test_labels = np.load(f'{self.dir_logits_labels}test_labels.npy')

    def _divide_classes_from_model(self):
        self.classes_train_logits = self._separate_logits_by_class(self.train_logits, self.train_labels)
        self.classes_test_logits = self._separate_logits_by_class(self.test_logits, self.test_labels)

    def _separate_logits_by_class(self, logits, labels):
        logits = np.array(logits).squeeze()
        labels = np.array(labels)

        unique_classes = np.unique(labels)
        return {cls: logits[labels == cls] for cls in unique_classes}

    def run_analysis(self):
        # Generate histograms for logits and likelihoods
        self.analysis.generate_histograms(self.classes_test_logits, 'logit', '1')

        likelihoods = {cls: self.analysis.logits_to_likelihoods(logits) for cls, logits in self.classes_test_logits.items()}
        self.analysis.generate_histograms(likelihoods, 'likelihood', '2')

        # Using KDE approach
        kde = sklearnKDE(self.classes_train_logits, self.test_logits)
        posterior_probs = kde.compute_posterior_prob()

        # Evaluate baseline with KDE
        self.analysis.compute_metrics(self.test_logits, posterior_probs, self.test_labels, 'METRICS IN TEST TIME')

if __name__ == "__main__":
    analysis = Analysis()
    post_process = PostProcessing(analysis)
    post_process.run_analysis()
    print("FINISHED!!")
