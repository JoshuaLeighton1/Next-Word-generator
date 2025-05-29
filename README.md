# Next Word Prediction

This project implements a next-word prediction model using a LSTM (Long Short-Term Memory) Neural network built with PyTorch. The model is trained on a text corpus to allow for prediction for the next word(s) in a sequence with debugging capabilities.

## Features

- LSTM-Based Model: Utilizes a multi-layer LSTM architecture for sequence modeling.
- Text Preprocessing: Handles text cleaning, vocabiulary building and sequence creation with support for special tokens "(< UNK >, < PAD >, < START >, "< END >") 
to help syntactical structure.
- Comprehensive Debugging: Text generation includes detailed debugging output to trace model predictions.
- Device Support: Automatically selects GPU(CUDA/MPS) or CPU based on availability.
- Evaluation Metrics: Reports test loss and accuracy for model performance assessment.

## Requirements

- Pytho 3.8+
- PyTorch
- NumPy
- Matplotlib
- Optional: A text corpus file or simple default text will be used

## Installation: 
1. Clone the repository:

``` 
git clone https://github.com/your-username/next-word-prediction.git

cd next-word-prediction
```

2. Install Dependencies:
``` 
pip install requirements.txt 
```

3. Prepare Data(Optional): Plce your text corpus file in the project root directory. If no file is provided, the model will use a default sample text.

## Usage

Run the main script to train the model and generate text predictions:  

``` 
python nextword.py
````

### What to Expect

1. Data Loading and Preprocessing:

- Loads the text corpus or uses a default sample.
- Cleans text by removing punctuation and converting to lowercase.
- Builds a vocabulary with a minimum word frequency threshold (default: 3).
- Creates sequences of 10 words for training.

2. Training: 

- Trains the LSTM model for 30 epochs with a batch size of 64.
- Uses Adam optimizer with a learning rate scheduler.
- Reports training and validation loss per epoch.

3. Evaluation: 
- Evaluates the model on a test set, reporting test loss and accuracy.

4. Text Generation:

- Generates text using the trained model with debugging output.
- Tests multiple sequences of common words with a temperature of 0.8.
- Includes timeout protection to prevent long-running generation.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

### Example output:

>Using device: cpu  
Loaded text with 123456 characters  
Preprocessing text...   
Vocabulary size: 5000  
Total sequences: 10000  
Training sequences: 7000  
Validation sequences: 1500  
Test sequences: 1500  
Model parameters: 2,345,678  
Epoch 1, Train Loss: 6.5432, Val Loss: 6.1234  
...  
Test Loss: 5.9876  
Test Accuracy: 0.2345 (23.45%)  
...  
DEBUGGING TEXT GENERATION  
Start words: ['the', 'quick', 'brown']  
...  
Final result: the quick brown fox jumps over the lazy dog and  



## Acknowledgements
- Built with PyTorch

- Inspired by common next-word prediction tasks in natural language processing.



