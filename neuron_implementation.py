# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# NPNeuron Implementation

# +
import numpy as np

class MPNeuron:
    def __init__(self):
        self.threshold = None # es  un número que se define de manera manual para identificar la cantidad de entradas que deben de estar activas
    
    def model(self, entry):
        # se procesan las caracteristicas de entrada
        # entry: [1, 0, 1, 0] [x1, x2, x3, x4, ..., xn]
        z = sum(entry) # función de agregación
        return z >= self.threshold
    
    def predict(self, entries):
        # entries : [[1, 0, 1, 0], [1, 0, 1, 0]]
        results = []
        for entry in entries:
            result = self.model(entry)
            results.append(result)
        return np.array(results)
    

# -

mp_neuron = MPNeuron()
mp_neuron.threshold = 3

# +
entries = [
    [1, 0, 0],
    [1, 1, 1],
    [0, 1, 1]
]



# -

if __name__ == '__main__':
    mp_neuron.predict(entries)


