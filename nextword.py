import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import random
import string


#Load data corpus

with open('corpus.txt', 'r') as file:
    text = file.read()


#remove punctuation and convert to lowercase 
text = text.translate(str.maketrans('', '', string(punctuation)))
text = text.lower()

#split into words to tokenize
words = text.split()
