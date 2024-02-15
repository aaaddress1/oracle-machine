# https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=EcVIDWAZEtjN&uniqifier=1
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print("length of dataset in characters: ", len(text))
    print(text[:1000])

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[0:block_size+0]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[ i+0 : i+block_size+0 ] for i in ix])
    y = torch.stack([data[ i+1 : i+block_size+1 ] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

print(xb) # our input to the transformer

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # e.g. (B=4, T=8, C=65)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx) # logits = (B, T, C)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(5000): # increase number of steps for good results... 
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
r'''
PS C:\Users\aaaddress1\Desktop\gptFromZeroToHero> Measure-Command{py .\biGramLM.py | Write-Host}
   ...
   
TbesShire tearor-tam de
FRIs woCEs Yeandst h lldolor g brit 'd yUTq-lk&ven'-uth t, ve!
Rak theto ty ond therewNASbe h s&d
ASkr h, lpMpealousrtomellfibeporthociO, ilddod,
O:
IVYCasn t hore br fore
CIA sis tuly gek makithONI'e oth, athajumphe fing ie
We nd
Werdy ly fin ento, meston, nTharq-ck.
ENoorindscqverQ-gadayVernd,&Boux?I'd, k e tghentyowilenet imerg l--mof W'd iathe!

INLin y I&on ot, w tritwe s sim ow rderingemes de therer dedes mf--tothe.
SVINCKUSim, wofe thampu tatpll?K:
A IJUNAURUER:

ThakaSe as f A-vefk rthishAy t therQulinERKITblooocooreay hul y d-d the!

ATTIUptor tooms llire My mvemn Be
Ser y;K:rdmurmbun

WASfubenJUElly t y BANGBOXE richave
I, mofoff zas, IUForse. reenangdadindenxEOreasldum,


e
Yolfffitld, bl'ere er:
BLenEr w
Se, WWhes jFYNapay tethore!uE:
GLBedofubl'VOKuit!
Sit LINCL:
IZKII R.
tr han:
Theaseise athicof
LOLBus CERYde tDesula tttithe lour wes kisasiseaUTw?Oinounera tlantUFKGave
IBFr nein t;AUBo p$s
I& 'switoud su, r t ide
xxatqe f boo te tt, ho morved'WAce


Days              : 0
Hours             : 0
Minutes           : 0
Seconds           : 13
Milliseconds      : 26
Ticks             : 130266817
TotalDays         : 0.000150771778935185
TotalHours        : 0.00361852269444444
TotalMinutes      : 0.217111361666667
TotalSeconds      : 13.0266817
TotalMilliseconds : 13026.6817
'''