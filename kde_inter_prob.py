import numpy as np
from sklearn.neighbors import KernelDensity

class BinaryKDE:
    def __init__(self, classes_logits_train, logits_test, kernel='gaussian', bandwidth=0.5):
        self.logits_pos, self.logits_neg = classes_logits_train.values()
        self.kde_pos = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(self.logits_pos.reshape(-1, 1))
        self.kde_neg = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(self.logits_neg.reshape(-1, 1))

        self.logits_test = logits_test
    
    def compute_posterior_prob(self, epsylon = 1e-7):
        # Log-likelihoods
        log_probs_pos = self.kde_pos.score_samples(self.logits_test.reshape(-1, 1))  
        log_probs_neg = self.kde_neg.score_samples(self.logits_test.reshape(-1, 1)) 

        likelihoods_pos = np.exp(log_probs_pos)
        likelihoods_neg = np.exp(log_probs_neg)
        
        return (likelihoods_pos + epsylon) / ((likelihoods_pos + epsylon) + (likelihoods_neg + epsylon))

    
