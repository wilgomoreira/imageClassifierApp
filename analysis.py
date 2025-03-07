import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from netcal.metrics import ECE
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, confusion_matrix

class Analysis:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_histograms(self, data_matrix, name, identifier, plot_dir='plots_and_histograms', bins=30):
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for cln in range(data_matrix.shape[1]):
            plt.hist(data_matrix[:, cln], bins=bins, histtype='step', linewidth=1.5, label=f'Class {cln}', density=True)

        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {name}')
        plt.savefig(f"{plot_dir}/{identifier}_{name}_histogram.pdf")
        plt.close()

    def compute_metrics(self, logits, enhanced_probs, true_labels, name):
        # Baseline
        baseline_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        baseline_metrics = self.calculate_metrics(true_labels, baseline_probs)
        
        # Enhanced approach
        enhanced_metrics = self.calculate_metrics(true_labels, enhanced_probs)
        
        # salve everything
        self.format_and_save_results(name, baseline_metrics, enhanced_metrics)
 
    def calculate_metrics(self, true_labels, preds, bins=10):
        max_probs, _ = torch.max(torch.from_numpy(preds), dim=1) 
        preds_labels = torch.argmax(torch.from_numpy(preds), dim=1)
        tn, fp, fn, tp = confusion_matrix(true_labels, preds_labels.numpy()).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        return {
            'accuracy': accuracy_score(true_labels, preds_labels.numpy()),
            'f1_score': f1_score(true_labels, preds_labels.numpy()),
            'avg_precision': average_precision_score(true_labels, max_probs.numpy()),
            'ece': ECE(bins=bins).measure(max_probs.numpy(), true_labels),
            'fpr': fpr,
            'fnr': fnr
        }

    def format_and_save_results(self, name, baseline, enhanced):
        text = (f'{name}\n'
                f'BASELINE:\n'
                f'ACC: {self.percent_format(baseline["accuracy"])}% | '
                f'F1 SCORE: {self.percent_format(baseline["f1_score"])}% | '
                f'AVG PREC: {self.percent_format(baseline["avg_precision"])}% | '
                f'ECE: {self.percent_format(baseline["ece"])}% | '
                f'FPR: {self.percent_format(baseline["fpr"])}% | '
                f'FNR: {self.percent_format(baseline["fnr"])}%\n'
                f'NEW APPROACH:\n'
                f'ACC: {self.percent_format(enhanced["accuracy"])}% | '
                f'F1 SCORE: {self.percent_format(enhanced["f1_score"])}% | '
                f'AVG PREC: {self.percent_format(enhanced["avg_precision"])}% | '
                f'ECE: {self.percent_format(enhanced["ece"])}% | '
                f'FPR: {self.percent_format(enhanced["fpr"])}% | '
                f'FNR: {self.percent_format(enhanced["fnr"])}%\n')

        with open(os.path.join(self.output_dir, f'{name.lower()}.txt'), 'w') as f:
            f.write(text)
    
    def percent_format(self, number, round_number=2):
        return round(100 * number, round_number)