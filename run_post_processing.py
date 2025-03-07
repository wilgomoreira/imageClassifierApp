import os
import numpy as np
from kde_inter_prob import sklearnKDE
from analysis import Analysis
import torch

class PostProcessing:
    def __init__(self, new_approach=sklearnKDE, analysis=Analysis, dir_logits_labels='logits_labels/'):
        self.analysis = analysis()
        self.new_approach = new_approach
        self.dir_logits_labels = dir_logits_labels

        self.train_logits, self.train_labels, self.test_logits, self.test_labels = self._load_data()

    def _load_data(self):
        os.makedirs(self.analysis.output_dir, exist_ok=True)
        os.makedirs(self.dir_logits_labels, exist_ok=True)

        return (np.load(f'{self.dir_logits_labels}train_logits.npy'),
                np.load(f'{self.dir_logits_labels}train_labels.npy'),
                np.load(f'{self.dir_logits_labels}test_logits.npy'),
                np.load(f'{self.dir_logits_labels}test_labels.npy'))

    def run_analysis(self):
        # Generate histograms for logits and likelihoods
        self.analysis.generate_histograms(self.test_logits, 'logit', '1')

        likelihoods = torch.softmax(torch.tensor(self.test_logits), dim=1).numpy()
        self.analysis.generate_histograms(likelihoods, 'baseline-probability', '2')

        # Using KDE approach
        kde = self.new_approach(self.train_logits, self.test_logits)
        posterior_probs = kde.compute_posterior_prob()
        self.analysis.generate_histograms(posterior_probs, 'KDE-probability', '3')

        # Evaluate baseline with KDE
        self.analysis.compute_metrics(self.test_logits, posterior_probs, self.test_labels, 'METRICS_BASELINE_VS_NEW_APPROACH')

def main():
    post_process = PostProcessing()
    post_process.run_analysis()
    print("FINISHED!!")

if __name__ == "__main__":
    main()
    
