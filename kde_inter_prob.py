import numpy as np
from sklearn.neighbors import KernelDensity
import math
import torch

class sklearnKDE:
    logits_pos: np
    logits_neg: np
    kde_pos: KernelDensity
    kde_neg: KernelDensity
    logits_test: np

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
    
class misKDE:
    logits_pos: np
    logits_neg: np
    kde_pos: KernelDensity
    kde_neg: KernelDensity
    logits_test: np

    def __init__(self, classes_logits_train, logits_test):
        self.logits_pos_train, self.logits_neg_train = classes_logits_train.values()
        self.logits_test = logits_test
        self.values_pos, self.values_neg = self._obtain_densities()
        self.like_pos, self.like_neg = self._obtain_likelihoods(self.values_pos, self.values_neg)

    def _obtain_densities(self, point_in_x=15):
        min_value = math.ceil(min(self.logits_pos_train.min(), self.logits_neg_train.min()))
        max_value = math.ceil(max(self.logits_pos_train.max(), self.logits_neg_train.max()))
        self.x_values = np.linspace(min_value, max_value, point_in_x) 
        
        dens_values_pos = np.array([self._kde(x, self.logits_pos_train) for x in self.x_values])
        dens_values_neg = np.array([self._kde(x, self.logits_neg_train) for x in self.x_values])

        cdf_values_pos = np.cumsum(dens_values_pos)
        cdf_values_pos = (cdf_values_pos - cdf_values_pos.min()) / (cdf_values_pos.max() - cdf_values_pos.min())

        cdf_values_neg = np.cumsum(dens_values_neg)
        cdf_values_neg = (cdf_values_neg - cdf_values_neg.min()) / (cdf_values_neg.max() - cdf_values_neg.min())
        cdf_values_neg = 1 - cdf_values_neg

        return dens_values_pos, dens_values_neg

    def _kde(self, x, data):
        n = len(data)
        sigma = np.std(data)
        h = (4 * sigma**5 / (3 * n))**(1/5)

        result = 0
        for xi in data:
            result += self._gaussian_kernel((x - xi) / h)
        
        normalization_factor = (n * h)
        if normalization_factor > 0:
            return result / normalization_factor
        else:
            return 0  

    def _gaussian_kernel(self, u):
        return (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * u**2)

    def _epanechnikov_kernel(self, u):
        return 0.75 * (1 - u**2) if abs(u) <= 1 else 0
    
    def _obtain_likelihoods(self, value_pos, value_neg):
        like_pos_test = self._linear_interpolation(self.x_values, value_pos, self.logits_test)
        like_neg_test = self._linear_interpolation(self.x_values, value_neg, self.logits_test)
        return like_pos_test, like_neg_test

    def _linear_interpolation(self, x_values, y_values, x_test, epsylon = 1e-7):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Convert to tensors and send to GPU
        x_values = torch.tensor(x_values, device=device, dtype=torch.float32)
        y_values = torch.tensor(y_values, device=device, dtype=torch.float32)
        x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
        
        # For each value in x_test, find the two indices in x_values ​​that it lies between
        indices = torch.searchsorted(x_values, x_test)

        # Fix indexes to handle limits
        indices = torch.clamp(indices, 1, len(x_values) - 1)

        # Get the two points of x and y that surround each point of x_test
        x0 = x_values[indices - 1]
        x1 = x_values[indices]
        y0 = y_values[indices - 1]
        y1 = y_values[indices]

        # Vectorized linear interpolation
        weight = (x_test - x0) / (x1 - x0)
        interpolated_values = y0 + weight * (y1 - y0)
        interpolated_values = torch.clamp(interpolated_values, 0, 1)

        # Apply Laplace smoothing
        interpolated_values = interpolated_values + epsylon

        return interpolated_values.cpu().numpy()

    def compute_posterior_prob(self, prior_pos=0.5, prior_neg=0.5, epsilon = 1e-3):
        evidence = (self.like_pos + epsilon) * prior_pos + (self.like_neg + epsilon) * prior_neg
        posterior_prob = (self.like_pos + epsilon) * prior_pos / evidence
        return posterior_prob

    
