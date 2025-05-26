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


#convert sentences to indices
sequences_idx=[]
for seq, target in sequences:
    #assigns the words index to the seq index
    seq_idx = [word_to_idx[word] for word in seq]
    target_idx = word_to_idx[target]
    #append to sequences array
    sequences_idx.append((seq_idx, target_idx))


# Create a PyTorch dataset

class TextDataSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __get_item__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(target)

#Create DataLoader for batching 

#pass the sequences_idx array as an arg for the TextDataSet Object
dataset = TextDataSet(sequences_idx)
batch_size = 32
#DataLoader represents an iterable dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Define the RNN Model

#Module is the base class for all neural networks
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        #convert word indices to embeddings
        embedded = self.embedding(x)
        #RNN processes the sequence
        output, hidden = self.rnn(embedded)
        #Use the last time steps output
        last_output = output[:, -1, :]
        #predict the next word
        prediction = self.fc(last_output)
        return prediction
    



    