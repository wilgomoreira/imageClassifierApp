import numpy as np
from netcal.metrics import ECE

# Simulando probabilidades previstas para a classe positiva
y_prob = np.array([0.99, 0.80, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.01])
# Rótulos verdadeiros
y_true = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0])

# Inicializando o ECE com 10 bins
ece = ECE(bins=10)

# Calculando o ECE
ece_score = ece.measure(y_prob, y_true)

print(f"Expected Calibration Error (ECE): {ece_score:.4f}")
