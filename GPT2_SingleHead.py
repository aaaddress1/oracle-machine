import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embd = 128
n_layer = 4
dropout = 0.1
# ------------

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        v = self.value(x)
        out = wei @ v
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SingleHeadCausalSelfAttention(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # pre-LN + residual (GPT-2 style)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SingleHeadGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # GPT-2 commonly ties token embedding with output projection.
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx


model = SingleHeadGPT2().to(device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.3f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=600)[0].tolist()))


'''
0.808M parameters
step 0: train loss 82.8554, val loss 82.5055
step 200: train loss 3.2480, val loss 3.2990
step 400: train loss 2.8768, val loss 2.9265
step 600: train loss 2.7393, val loss 2.7760
step 800: train loss 2.6499, val loss 2.6835
step 1000: train loss 2.5927, val loss 2.5966
step 1200: train loss 2.5181, val loss 2.5345
step 1400: train loss 2.4547, val loss 2.4605
step 1600: train loss 2.3753, val loss 2.3703
step 1800: train loss 2.3502, val loss 2.3627
step 1999: train loss 2.2862, val loss 2.2942

SuES:
Sh the blichouthe,
I athome pot diant aen you thall vire hat, tha;
And lis is dourd shen,
Hor your ham ber Eu thore
Aay bros thour seare, hored,
Ands wit. Gom wilis me.
INA:
RonE
Lil--e, ll Me silo, yountish nong her burycringlcen be ig, thoughere, VI ray Iasll cr  out
Thile,
Ala han 'e!

ROLAUS:
And bue.
Bur atly sis mald s! conhe for an:
Sand youll, beh.u be shithest you thy bour hant wonget
AsE:
Ahe soom tis

ARDus the no withereand dean ch sh hishaorall king pryUs his ive,
HEES:
my iroanchamljjue ave ny at gits Gene thele thois hespaicant king!

RIUS:
Foike to rn menmin you you darer
'''