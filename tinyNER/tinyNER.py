# rewrite from standford CS230
# ref: https://cs230.stanford.edu/blog/namedentity/#problem-setup

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class Net(nn.Module):
    def __init__(self, vocab_size, number_of_tags):
        super(Net, self).__init__()
        EMBEDDING_DIM, LSTM_HIDDEN_DIM = 50, 50 
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(LSTM_HIDDEN_DIM, number_of_tags)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x seq_len x embedding_dim
        s = self.embedding(s)

        # run the LSTM along the sentences of length seq_len
        # dim: batch_size x seq_len x lstm_hidden_dim
        s, _ = self.lstm(s)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        # dim: batch_size*seq_len x lstm_hidden_dim
        s = s.view(-1, s.shape[2])

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))


if __name__ == "__main__":
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)

    words, tags =  Counter(
        open("dataset_small/train/sentences.txt").read().split() + \
        open("dataset_small/test/sentences.txt").read().split() + \
        open("dataset_small/val/sentences.txt").read().split()
    ), Counter(
        open("dataset_small/train/labels.txt").read().split() + \
        open("dataset_small/test/labels.txt").read().split() + \
        open("dataset_small/val/labels.txt").read().split()
    )
    
    # Add pad tokens
    # Hyper parameters for the vocab
    PAD_WORD, PAD_TAG, UNK_WORD = '<pad>', 'O', 'UNK' 
    
    # Only keep most frequent tokens
    MIN_COUNT = 1
    words = [tok for tok, count in words.items() if count >= MIN_COUNT ]
    tags = [tok for tok, count in tags.items() if count >= MIN_COUNT]

    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    
    # add word for unknown words 
    words.append(UNK_WORD)
    print(len(words), len(tags))

    model = Net( len(words), len(tags) )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    vocab, tag_map = { x: i for i, x in enumerate(words) }, { x: i for i, x in enumerate(tags) }

    # setting the indices for UNKnown words and PADding symbols
    unk_ind, pad_ind = vocab[UNK_WORD], vocab[PAD_WORD]

    sentences, labels = list(), list()
    for sentence in open("dataset_small/train/sentences.txt").read().splitlines():
        s = [ vocab[token] if token in vocab else unk_ind for token in sentence.split(' ') ]
        sentences.append(s)
    for sentence in open("dataset_small/train/labels.txt").read().splitlines():
        l = [ tag_map[label] for label in sentence.split(' ') ]
        labels.append(l)
    batch_max_len = max([len(s) for s in sentences])
    print("")

    def get_batch( sentence, tags ) -> torch.tensor:
        batch_data = pad_ind*np.ones( (1, batch_max_len) )
        batch_labels =  -1 * np.ones( (1, batch_max_len) )
        batch_data[0][:len(sentence)] = sentence
        batch_labels[0][:len(sentence)] = tags    
        data, labels = torch.tensor(batch_data, dtype=torch.long), torch.tensor(batch_labels, dtype=torch.long)
        return data, labels
    
    for _ in range(100): # epoch for 10 times
        for i in range(len(sentences)):
            train_batch, labels_batch = get_batch( sentences[i], labels[i] )
            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)


            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

        # Evaluate summaries only once in a while
        print(f"loss: {loss.item():.2f}")

    print("finished training!")
    for s_indx in range(10):
        train_batch, labels_batch = get_batch( sentences[s_indx], labels[s_indx] )

        output_batch = model(train_batch) # size = (30, 9)
        predict = [ torch.argmin(torch.abs( each_ner) ).tolist() for each_ner in output_batch ]
        print("guess:", predict)
        print("correct:", labels[s_indx] )
    
        #guess_ner = [t for t in tag_map if tag_map[t] == 0].pop()
        print(f"")