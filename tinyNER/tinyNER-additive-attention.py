import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Sample toy dataset
sentences = [
    ["John", "lives", "in", "New", "York"],
    ["Mary", "is", "from", "Los", "Angeles"],
]
tags = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC"],
    ["B-PER", "O", "O", "B-LOC", "I-LOC"],
]

# Vocabulary & tag mapping
word_to_ix = {word: i+1 for i, word in enumerate(set(word for sent in sentences for word in sent))}
word_to_ix["<PAD>"] = 0  # Padding token

tag_to_ix = {tag: i for i, tag in enumerate(set(tag for tag_seq in tags for tag in tag_seq))}
ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
ATTN_DIM = 64
BATCH_SIZE = 2
PAD_IDX = word_to_ix["<PAD>"]

# Convert sentences and tags to indices
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)

train_data = [(prepare_sequence(sent, word_to_ix), prepare_sequence(tag_seq, tag_to_ix)) for sent, tag_seq in zip(sentences, tags)]

# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Dataloader
train_loader = DataLoader(NERDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)

# **BiLSTM with Additive Attention Model**
class BiLSTMAttentionNER(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, attn_dim):
        super(BiLSTMAttentionNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Additive Attention
        self.attn_W = nn.Linear(hidden_dim * 2, attn_dim)  # W_a
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)   # v_a
        
        # Classifier
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
    
    def attention(self, lstm_output):
        """ Apply Additive Attention on LSTM outputs """
        attn_weights = torch.tanh(self.attn_W(lstm_output))  # (batch, seq_len, attn_dim)
        attn_weights = self.attn_v(attn_weights)             # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)    # Normalize
        context = attn_weights * lstm_output                 # Apply attention weights
        context = context.sum(dim=1)                         # Sum over sequence length
        return context
    
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        context = self.attention(lstm_out)
        tag_scores = self.fc(lstm_out)  # Predict tags at each timestep
        return tag_scores

# Model instance
model = BiLSTMAttentionNER(vocab_size=len(word_to_ix), tagset_size=len(tag_to_ix),
                           embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    for words, tags in train_loader:
        optimizer.zero_grad()
        outputs = model(words)
        
        outputs = outputs.view(-1, len(tag_to_ix))  # Flatten for loss calculation
        tags = tags.view(-1)  # Flatten target
        loss = criterion(outputs, tags)

        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Prediction Example
def predict(sentence):
    with torch.no_grad():
        inputs = prepare_sequence(sentence, word_to_ix).unsqueeze(0)  # Add batch dimension
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=-1).squeeze().tolist()
        return [ix_to_tag[idx] for idx in predictions]

print("\nPredictions:", predict(["John", "lives", "in", "New", "York"]))
