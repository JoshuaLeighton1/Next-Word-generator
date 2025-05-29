import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import random
import string
import matplotlib.pyplot as plt
from collections import Counter
import time
import sys
import threading
from threading import Timer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check for GPU availability
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global flag for timeout handling
generation_timeout = False

def timeout_handler():
    global generation_timeout
    generation_timeout = True
    print("\n⚠️  TIMEOUT: Text generation taking too long, stopping...")

# Load data corpus
try:
    with open('pg2680.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    print(f"Loaded text with {len(text)} characters")
except FileNotFoundError:
    # Create sample text for demonstration
    text = """
    the quick brown fox jumps over the lazy dog. the dog was sleeping under the tree.
    a bird flew over the house and landed on the roof. the cat watched the bird from below.
    in the morning the sun rises in the east. people wake up and start their day.
    children go to school while adults go to work. families gather in the evening.
    the weather is nice today and the birds are singing. flowers bloom in spring.
    water flows in the river and fish swim in the lake. trees grow tall in the forest.
    """ * 200  # More repetitions for better learning
    print("Using sample text for demonstration")

# Preprocessing
print("Preprocessing text...")
text = text.translate(str.maketrans('', '', string.punctuation))
text = text.lower()
words = text.split()

print(f"Total words before filtering: {len(words)}")

# Build vocabulary with even less aggressive filtering
word_counts = Counter(words)

# Keep more words to improve vocabulary coverage
min_word_freq = 3  # Keep almost all words
filtered_words = [word for word in words if word_counts[word] >= min_word_freq]

# Create a more manageable vocabulary by taking most common words
top_words = [word for word, count in word_counts.most_common()]  # Limit vocab size
vocab = set(top_words)

# Add special tokens
special_tokens = ['<UNK>', '<PAD>', '<START>', '<END>']
vocab.update(special_tokens)

# Create mappings
word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Most common words: {[word for word, _ in word_counts.most_common()]}")

# Handle unknown words
def word_to_index(word):
    return word_to_idx.get(word, word_to_idx['<UNK>'])

# Create sequences with better filtering
sequence_length = 10 # Even shorter sequences for better learning
sequences = []

print("Creating training sequences...")
for i in range(sequence_length, len(filtered_words)):  # Limit total sequences
    seq = filtered_words[i-sequence_length:i]
    target = filtered_words[i]
    
    # Only keep sequences where target is in our vocabulary
    if target in vocab:
        seq_idx = [word_to_index(word) for word in seq]
        target_idx = word_to_index(target)
        sequences.append((seq_idx, target_idx))

print(f"Total sequences: {len(sequences)}")

if len(sequences) == 0:
    print("❌ No valid sequences found! Check your data.")
    sys.exit(1)

# Split data
train_size = int(0.7 * len(sequences))
val_size = int(0.15 * len(sequences))

train_sequences = sequences[:train_size]
val_sequences = sequences[train_size:train_size + val_size]
test_sequences = sequences[train_size + val_size:]

print(f"Training sequences: {len(train_sequences)}")
print(f"Validation sequences: {len(val_sequences)}")
print(f"Test sequences: {len(test_sequences)}")

# Dataset class
class TextDataSet(Dataset):
    def __init__(self, sequences, device):
        self.sequences = sequences
        self.device = device

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long, device=self.device), \
               torch.tensor(target, dtype=torch.long, device=self.device)

# Create datasets and dataloaders
train_dataset = TextDataSet(train_sequences, device)
val_dataset = TextDataSet(val_sequences, device)
test_dataset = TextDataSet(test_sequences, device)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Simplified LSTM Model
class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=2):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

# Create model
model = SimpleLSTMModel(vocab_size).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

# Quick training (reduced epochs for testing)
num_epochs = 30
print("Starting quick training...")

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    num_batches = 0
    
    for seq, target in train_dataloader:
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = total_train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    total_val_loss = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for seq, target in val_dataloader:
            output = model(seq)
            loss = criterion(output, target)
            total_val_loss += loss.item()
            num_val_batches += 1
    
    avg_val_loss = total_val_loss / num_val_batches
    val_losses.append(avg_val_loss)
    
    scheduler.step()
    
    print(f'Epoch {epoch+1:2d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Test the model
print("\n" + "="*50)
print("EVALUATING MODEL")
print("="*50)

model.eval()
total_test_loss = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for seq, target in test_dataloader:
        output = model(seq)
        loss = criterion(output, target)
        total_test_loss += loss.item()
        
        _, predicted = torch.max(output, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

test_loss = total_test_loss / len(test_dataloader)
test_accuracy = correct_predictions / total_predictions

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ROBUST TEXT GENERATION WITH COMPREHENSIVE DEBUGGING
def generate_text_with_debugging(model, start_words, num_words=12, temperature=1.0):
    """
    Text generation with step-by-step debugging and timeout protection
    """
    global generation_timeout
    generation_timeout = False
    
    print(f"\n🔍 DEBUGGING TEXT GENERATION")
    print(f"Start words: {start_words}")
    print(f"Requested words: {num_words}")
    print(f"Temperature: {temperature}")
    
    # Set timeout timer
    timer = Timer(15.0, timeout_handler)  # 15 second timeout
    timer.start()
    
    model.eval()
    
    # Validate start words
    print(f"\n📝 Validating start words...")
    for i, word in enumerate(start_words):
        if word in word_to_idx:
            print(f"  {i+1}. '{word}' -> index {word_to_idx[word]} ✓")
        else:
            print(f"  {i+1}. '{word}' -> <UNK> (index {word_to_idx['<UNK>']}) ⚠️")
    
    # Convert to indices
    current_seq = [word_to_index(word) for word in start_words]
    generated_words = start_words.copy()
    
    print(f"\n🔄 Starting generation loop...")
    
    for step in range(num_words):
        if generation_timeout:
            print(f"⏰ Generation stopped due to timeout at step {step}")
            break
            
        try:
            print(f"\n--- Step {step + 1} ---")
            print(f"Current sequence indices: {current_seq}")
            print(f"Current sequence words: {[idx_to_word.get(idx, '<UNK>') for idx in current_seq]}")
            
            # Create input tensor
            seq_tensor = torch.tensor(current_seq, dtype=torch.long, device=device).unsqueeze(0)
            print(f"Input tensor shape: {seq_tensor.shape}")
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                output = model(seq_tensor)
                forward_time = time.time() - start_time
                
            print(f"Forward pass took {forward_time:.4f} seconds")
            print(f"Output shape: {output.shape}")
            
            # Check for invalid outputs
            if torch.isnan(output).any():
                print("❌ NaN detected in model output!")
                break
            if torch.isinf(output).any():
                print("❌ Infinity detected in model output!")
                break
            
            # Apply temperature and get probabilities
            logits = output / max(temperature, 0.1)
            probabilities = torch.softmax(logits, dim=1).squeeze()
            
            print(f"Probabilities shape: {probabilities.shape}")
            print(f"Probabilities sum: {probabilities.sum().item():.6f}")
            
            # Get top predictions for debugging
            top_k = 10
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            print(f"Top {top_k} predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                word = idx_to_word.get(idx.item(), '<UNK>')
                print(f"  {i+1}. '{word}' (idx: {idx.item()}) -> {prob.item():.4f}")
            
            # Sample next word (using top prediction for reliability)
            next_word_idx = top_indices[0].item()
            next_word = idx_to_word.get(next_word_idx, '<UNK>')
            
            print(f"Selected: '{next_word}' (index: {next_word_idx})")
            
            # Skip special tokens
            if next_word in special_tokens:
                print(f"Skipping special token: {next_word}")
                # Try second best prediction
                if len(top_indices) > 1:
                    next_word_idx = top_indices[1].item()
                    next_word = idx_to_word.get(next_word_idx, '<UNK>')
                    print(f"Using second choice: '{next_word}'")
                else:
                    continue
            
            generated_words.append(next_word)
            
            # Update sequence (sliding window)
            current_seq = current_seq[1:] + [next_word_idx]
            
            print(f"✓ Step {step + 1} completed successfully")
            
        except Exception as e:
            print(f"❌ Error at step {step + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            break
    
    timer.cancel()  # Cancel timeout timer
    
    result = ' '.join(generated_words)
    print(f"\n🎉 Generation completed!")
    print(f"Final result: {result}")
    
    return result

# TEXT GENERATION TESTING
print(f"\n" + "="*60)
print("STARTING COMPREHENSIVE TEXT GENERATION TESTING")
print("="*60)

# Test with most common words
common_words = [word for word, _ in word_counts.most_common(500) 
                if word in word_to_idx and word not in special_tokens]

print(f"Available common words: {common_words[100]}")

if len(common_words) >= 3:
    test_cases = [
        common_words[:30],
        common_words[30:80],
        common_words[80:100],
    ]
    
    for i, start_words in enumerate(test_cases):
        print(f"\n{'='*40}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*40}")
        
        try:
            result = generate_text_with_debugging(
                model, 
                start_words, 
                num_words=15,
                temperature=0.8
            )
            
            if result:
                print(f"\n✅ SUCCESS: {result}")
            else:
                print(f"\n❌ FAILED: No result returned")
                
        except Exception as e:
            print(f"\n❌ EXCEPTION in test case {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "-"*40)

else:
    print("❌ Not enough common words available for testing")

print(f"\n" + "="*60)
print("TEXT GENERATION TESTING COMPLETE")
print("="*60)

# Show final statistics
print(f"\nFinal Model Statistics:")
print(f"- Vocabulary size: {vocab_size}")
print(f"- Training sequences: {len(train_sequences)}")
print(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"- Final test accuracy: {test_accuracy:.2%}")
print(f"- Device used: {device}")