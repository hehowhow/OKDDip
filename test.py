import numpy as np

current = np.clip(100, 0.0, 80)
phase = 1.0 - 10 / 80
print(float(np.exp(-5.0 * phase * phase)))