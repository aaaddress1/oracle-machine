# https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=2Num7sX9CKOH&uniqifier=1
# total 2.14 mins for 1000 iters
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# -------------------------------------------------------------

# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

out.shape
#print(wei[0])
# ---------------------------------------------------------

k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
print((q @ k.transpose(-2, -1))[0])
print((q @ k.transpose(-2, -1) * head_size**-0.5)[0])
wei = q @ k.transpose(-2, -1) * head_size**-0.5

# -------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

#  context = torch.tensor( [ encode('hello') ] )
#  print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

import IPython
IPython.embed()
r'''
PS C:\Users\aaaddress1\Desktop\gptFromZeroToHero> Measure-Command{ py .\tinyNanoGPT.py | Write-Host} (5000 iters)
0.209729 M parameters
step 0: train loss 4.4116, val loss 4.4022
step 100: train loss 2.6568, val loss 2.6670
step 200: train loss 2.5090, val loss 2.5058
step 300: train loss 2.4196, val loss 2.4336
step 400: train loss 2.3502, val loss 2.3567
step 500: train loss 2.2964, val loss 2.3128
step 600: train loss 2.2410, val loss 2.2496
step 700: train loss 2.2060, val loss 2.2201
step 800: train loss 2.1639, val loss 2.1869
step 900: train loss 2.1233, val loss 2.1494
step 1000: train loss 2.1028, val loss 2.1301
step 1100: train loss 2.0705, val loss 2.1190
step 1200: train loss 2.0382, val loss 2.0790
step 1300: train loss 2.0248, val loss 2.0645
step 1400: train loss 1.9924, val loss 2.0369
step 1500: train loss 1.9711, val loss 2.0311
step 1600: train loss 1.9617, val loss 2.0468
step 1700: train loss 1.9399, val loss 2.0109
step 1800: train loss 1.9090, val loss 1.9946
step 1900: train loss 1.9079, val loss 1.9873
step 2000: train loss 1.8875, val loss 1.9996
step 2100: train loss 1.8710, val loss 1.9753
step 2200: train loss 1.8584, val loss 1.9607
step 2300: train loss 1.8551, val loss 1.9538
step 2400: train loss 1.8439, val loss 1.9473
step 2500: train loss 1.8174, val loss 1.9451
step 2600: train loss 1.8245, val loss 1.9395
step 2700: train loss 1.8133, val loss 1.9358
step 2800: train loss 1.8039, val loss 1.9237
step 2900: train loss 1.8061, val loss 1.9339
step 3000: train loss 1.7959, val loss 1.9193
step 3100: train loss 1.7673, val loss 1.9183
step 3200: train loss 1.7541, val loss 1.9126
step 3300: train loss 1.7562, val loss 1.9090
step 3400: train loss 1.7569, val loss 1.8954
step 3500: train loss 1.7367, val loss 1.8987
step 3600: train loss 1.7269, val loss 1.8873
step 3700: train loss 1.7282, val loss 1.8819
step 3800: train loss 1.7237, val loss 1.8930
step 3900: train loss 1.7232, val loss 1.8736
step 4000: train loss 1.7151, val loss 1.8665
step 4100: train loss 1.7157, val loss 1.8794
step 4200: train loss 1.7050, val loss 1.8599
step 4300: train loss 1.6993, val loss 1.8464
step 4400: train loss 1.7061, val loss 1.8613
step 4500: train loss 1.6888, val loss 1.8474
step 4600: train loss 1.6839, val loss 1.8317
step 4700: train loss 1.6816, val loss 1.8430
step 4800: train loss 1.6695, val loss 1.8500
step 4900: train loss 1.6685, val loss 1.8318
step 4999: train loss 1.6643, val loss 1.8226

ROMEO:
But you freight my sweeth. God untary;
If Norne Dom this fitter of where
while or that mustied to thyself
Whom I have alond you gliman; in a ceephant,
surs hance tongen he begave: not to by bett,
Virdom tow, go: bitt, Dittre so halls wifill my son and your sInaxh awo----
But too couldly: he but amt the but
sear
By Look's to; contraid,
that thrifted that grove to him, he's subort!
The kingle me; an your tought, for yet is
With the fear and littomes: is them,
But not will grave honour die forgiving:
With Morrow dierwn! doth as father scorn'd.

POMPERY:
Of be us.

LUCIO:
As, as lewsser:
hiscar: I ender and head which my talk my is fortuny,
What make speak the rect to thy subst
That us beltrs-dear.

KING-
FRUTHESS OF YORK:
I no Butunt! from where rivioly priceliong to-divings more in?

PRINTANUS:
Intrightrages.
They wrating, I'll bear confend thyself, I at and herells?

DUS:
It you, as is mistraid make dight
Frientage: thou, name a wouldsts.

Mown Plage,
Where him my you
That thou broy;
And in dentry, weinsuil on onjur
My men volincy, fortune, this pantar.

TLPETEO:
What squesch druwn, lettle.

PUARDINA:
Sweell addowell to prower!
Lady, whithon my worsomes? Enswomfult now my your bove give.
What that, our prople; blead you heave it.

SDLINGBRALUS:
Her.
Your bout to not: that thou fult,
An kingerliently but your gone, how offerds frue,
That madeed scrow steed, it you:
Onfull him it no arprand! I play's,
If not sudle fetch unto will rest:
My brive, here lever and made the fortable took.
Tust we you are flamer align.

MENEN:
Vew, tondus bow forgentle made yoour never.

KING RICHARD II:
Who too nead?

LORD GSUM:
Nay, that brird other.

All:
We hirsce.

MONTAM:
You he;
O, you to
melle'd they us thus us priet with requmationuo,
To the bretcius unforte that I would back withinds.

MOWIDIUS:
How, is all morre.
If or yourt, that prest to is was thus guet.
Plafenmer'd, no sure dreads in them,
At nocly that it to recondly
To canter to you achip in Con to suls;
For I know to


Days              : 0
Hours             : 0
Minutes           : 8
Seconds           : 16
Milliseconds      : 743
Ticks             : 4967430604
TotalDays         : 0.00574934097685185
TotalHours        : 0.137984183444444
TotalMinutes      : 8.27905100666667
TotalSeconds      : 496.7430604
TotalMilliseconds : 496743.0604
'''