import torch
import numpy as np
import os
from ase import Atoms

qm9tut = 'test/qm9tut'
best_model = torch.load(os.path.join(qm9tut, 'best_inference_model'), map_location=torch.device('cpu'))
print(best_model)