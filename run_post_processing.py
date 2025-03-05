import os
import numpy as np
from kde_inter_prob import sklearnKDE
from analysis import Analysis

class PostProcessing:
    analysis: Analysis
    dir_logits_labels: str
    train_logits: np
    train_labels: np
    test_logits: np
    test_labels: np
    train_logits_all_cl: dict
    test_logits_all_cl: dict

    def __init__(self, new_approach=sklearnKDE, analysis=Analysis, dir_logits_labels='logits_labels/'):
        self.analysis = analysis()
        self.new_approach = new_approach
        self.dir_logits_labels = dir_logits_labels

        self.train_logits, self.train_labels, self.test_logits, self.test_labels = self._load_data()
        self.train_logits_all_cl, self.test_logits_all_cl = self._divide_classes_from_model()

    def _load_data(self):
        os.makedirs(self.analysis.output_dir, exist_ok=True)
        os.makedirs(self.dir_logits_labels, exist_ok=True)

        return (np.load(f'{self.dir_logits_labels}train_logits.npy'),
                np.load(f'{self.dir_logits_labels}train_labels.npy'),
                np.load(f'{self.dir_logits_labels}test_logits.npy'),
                np.load(f'{self.dir_logits_labels}test_labels.npy'))

    def _divide_classes_from_model(self):
        train_logits_all_cl = self._separate_logits_by_class(self.train_logits, self.train_labels)
        test_logits_all_cl = self._separate_logits_by_class(self.test_logits, self.test_labels)
        return train_logits_all_cl, test_logits_all_cl

    def _separate_logits_by_class(self, logits, labels):
        logits = np.array(logits).squeeze()
        labels = np.array(labels)

        unique_classes = np.unique(labels)
        return {cls: logits[labels == cls] for cls in unique_classes}

    def run_analysis(self):
        # Generate histograms for logits and likelihoods
        self.analysis.generate_histograms(self.test_logits_all_cl, 'logit', '1')

        likelihoods = {cls: self.analysis.logits_to_likelihoods(logits) for cls, logits in self.test_logits_all_cl.items()}
        self.analysis.generate_histograms(likelihoods, 'likelihood', '2')

        # Using KDE approach
        kde = self.new_approach(self.train_logits_all_cl, self.test_logits)
        posterior_probs = kde.compute_posterior_prob()

        # Evaluate baseline with KDE
        self.analysis.compute_metrics(self.test_logits, posterior_probs, self.test_labels, 'METRICS IN TEST TIME')

def main():
    post_process = PostProcessing()
    post_process.run_analysis()
    print("FINISHED!!")

if __name__ == "__main__":
    main()
    
