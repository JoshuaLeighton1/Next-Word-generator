import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import random
import string
import matplotlib.pyplot as plt


#set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"using device {device}")



#Load data corpus
try:
    with open('pg2680.txt', 'r') as file:
        text = file.read()
except FileNotFoundError:
    raise FileNotFoundError("the file was not found. Please ensure it exists in the working directory")
except Exception as e:
    raise Exception(f"Error reading file {str(e)}")




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
sequence_length = 8
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

#split into training and validation sets (80/20)
train_size = int(0.8 * len(sequences_idx))
train_sequences = sequences_idx[:train_size]
val_sequences = sequences_idx[train_size:]


# Create a PyTorch dataset

class TextDataSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq), torch.tensor(target)

#Create DataLoader for batching 

#pass the sequences_idx array as an arg for the TextDataSet Object
dataset = TextDataSet(sequences_idx)
batch_size = 64
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
    
#set hyper parameters

embedding_dim = 200
hidden_size =  256
model = RNNModel(vocab_size, embedding_dim, hidden_size)

#train with CrossEntropyLoss function for classification
criterion = nn.CrossEntropyLoss()
#Adam optimizer with learning rate 0.001
optimizer =optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30
#track losses for plotting
epoch_losses=[]

for epoch in range(num_epochs):
    model.train()   
    total_loss = 0
    for seq, target in dataloader:
        seq, target = seq.to(device), target.to(device)
        #reset gradients
        optimizer.zero_grad()
        #forward pass
        output = model(seq)
        #compute loss
        loss = criterion(output, target)
        #Use back progagation 
        loss.backward()
        #update weights
        optimizer.step()
        total_loss += loss.item()
    #calculate average loss to be total loss / length of the dataset iterable
    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')

#Plot loss function 
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), epoch_losses, '-b', label='Training loss')
plt.title('Training loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('loss_plot.png')

#generate text method
def generate_text(model, start_seq, num_words):
    model.eval()
    current_seq = start_seq.copy()
    generated = current_seq.copy()
    #Validate start_seq
    for word in current_seq:
        if word not in word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary")

    for _ in range(num_words):
        seq_tensor = torch.tensor([word_to_idx[word] for word in current_seq], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(seq_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze()
        #sample next word
        next_word_idx = torch.multinomial(probabilities, 1).item()
        next_word = idx_to_word[next_word_idx]
        generated.append(next_word)
        #Add the next word into current sequence
        current_seq = current_seq[1:] + [next_word]
    
    return ' '.join(generated)


try: 
    start_seq = ['i', 'am', 'a', 'human', 'and']
    generated_text = generate_text(model, start_seq, 15)
    print(f"generated_text: {generated_text}")

except ValueError as e:
    print(f"Error: {str(e)}")



    