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
text = text.translate(str.maketrans('', '', string.punctuation))
text = text.lower()

#split into words to tokenize
words = text.split()

#build vocabulary: map words to unique indices
vocab = set(words)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)
#create sequence for training

#number of words in each input sentence
sequence_length = 5 
sequences = []
for i in range(sequence_length, len(words)):
    seq = words[i-sequence_length:i]
    target=words[i]
    sequences.append((seq, target))




