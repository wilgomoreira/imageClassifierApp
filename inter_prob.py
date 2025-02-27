import numpy as np
from sklearn.neighbors import KernelDensity

KERNEL = 'gaussian'
BANDWIDTH = 0.5

class BinaryKDE:
    def __init__(self, classes_logits_train, logits_test):
        self.logits_pos, self.logits_neg = classes_logits_train.values()
        self.kde_pos = KernelDensity(kernel=KERNEL, bandwidth=BANDWIDTH).fit(self.logits_pos.reshape(-1, 1))
        self.kde_neg = KernelDensity(kernel=KERNEL, bandwidth=BANDWIDTH).fit(self.logits_neg.reshape(-1, 1))

        self.logits_test = logits_test
        self.compute_posterior()

    
    def compute_posterior(self):
        # Log-likelihoods
        log_probs_pos = self.kde_pos.score_samples(self.logits_test.reshape(-1, 1))  
        log_probs_neg = self.kde_neg.score_samples(self.logits_test.reshape(-1, 1)) 

        likelihoods_pos = np.exp(log_probs_pos)
        likelihoods_neg = np.exp(log_probs_neg)

        evidence = likelihoods_pos + likelihoods_neg
        evidence = np.where(evidence == 0, 1e-10, evidence)

        self.posterior_probs = likelihoods_pos / evidence
    
