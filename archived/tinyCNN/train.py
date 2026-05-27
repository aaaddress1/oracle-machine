# Stanford CS230 
# ref: https://cs230.stanford.edu/blog/pytorch/
# ref: https://cs230.stanford.edu/blog/handsigns/
import random
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

USE_CUDA = False

class Net(nn.Module):
    def __init__(self):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        NUM_CHANNELS, DROPOUT_RATE = 32, 0.8
        self.num_channels = NUM_CHANNELS

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 6)       
        self.dropout_rate = DROPOUT_RATE

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


class SIGNSDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]

def fetch_dataloader(types, data_dir):


    dataloaders = {}
    # borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    # and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # define a training image loader that specifies transforms on images. See documentation for more details.
    train_transformer = transforms.Compose([
        transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor()])  # transform it into a torch tensor

    # loader for evaluation, no horizontal flip
    eval_transformer = transforms.Compose([
        transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
        transforms.ToTensor()])  # transform it into a torch tensor

    BATCH_SIZE, NUM_WORKERS = 32, 4
    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=BATCH_SIZE, shuffle=True,
                                        num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
            else:
                dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=USE_CUDA)

            dataloaders[split] = dl
    return dataloaders





def evaluate(model, loss_fn, dataloader):
    # summary for current eval loop
    collect_acc, collect_loss = [], []

    def accuracy(outputs, labels):
        outputs = np.argmax(outputs, axis=1)
        return np.sum(outputs==labels)/float(labels.size)


    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        #if params.cuda:
        #    data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        acc_val = accuracy(output_batch, labels_batch)
        collect_acc.append(acc_val)
        collect_loss.append(loss.item())
    print( f"accuracy: {np.mean(collect_acc):.2f}; loss: {np.mean(collect_loss):.2f}" )


if __name__ == "__main__":

    DATA_DIR = "C:/Users/aaadd/cs230-code-examples/pytorch/vision/data/64x64_SIGNS"
    dataloaders = fetch_dataloader( ['train', 'val'], DATA_DIR)
    train_dl, val_dl = dataloaders['train'], dataloaders['val']

    model = Net().cuda() if USE_CUDA else Net()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)

    # Train the model

    for epoch in range(10):

        loss_collect = []
        # Use tqdm for progress bar
        with tqdm(total=len(train_dl)) as t:
            for i, (train_batch, labels_batch) in enumerate(train_dl):
                # move to GPU if available
                #if params.cuda:
                #    train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
                
                # compute model output and loss
                output_batch = model(train_batch)
                loss = loss_fn(output_batch, labels_batch)

                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                '''
                if i % 100 == 0:
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()
                '''
                # update the average loss
                loss_collect.append(loss.item())

                t.set_postfix(loss='{:05.3f}'.format(  sum(loss_collect) / len(loss_collect) ))
                t.update()
        evaluate(model, loss_fn, train_dl)