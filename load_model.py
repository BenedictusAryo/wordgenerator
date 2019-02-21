
## Test Load rnn model


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# Load model



                                                                                                                          
# Open pretrained model                                                                                    
with open('rnn_x_epoch.net', 'rb') as f:
        checkpoint = torch.load(f)
            
loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])


# Sample using a loaded model
print(sample(loaded, 2000, top_k=5, prime="I go to school"))



