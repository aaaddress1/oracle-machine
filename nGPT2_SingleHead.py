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
dropout = 0.0
eps = 1e-6
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


def l2_normalize(x):
    return x * torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def normalized_lerp(a, b, alpha):
    # nGPT-style residual update on unit sphere: normalize -> interpolate -> normalize.
    a = l2_normalize(a)
    b = l2_normalize(b)
    out = a + alpha * (b - a)
    return l2_normalize(out)


@torch.no_grad()
def estimate_loss(model):
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


class NSingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # learnable scaling for normalized q/k (initialized so effective scale ~= 1)
        self.base_scale = n_embd**-0.5
        self.sqk = nn.Parameter(self.base_scale * torch.ones(n_embd))

    def forward(self, x):
        _, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        sqk = torch.abs(self.sqk).view(1, 1, C) * (C**0.5)
        q = l2_normalize(q) * sqk
        k = l2_normalize(k) * sqk

        # nGPT uses normalized q/k and larger softmax scale.
        wei = (q @ k.transpose(-2, -1)) * (C**0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class SwiGLUMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 8 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.base_scale = n_embd**-0.5
        self.suv = nn.Parameter(self.base_scale * torch.ones(8 * n_embd))

    def forward(self, x):
        uv = self.c_fc(x)
        # Same spirit as nGPT: scale gate/value stream before SwiGLU.
        suv = torch.abs(self.suv).view(1, 1, -1) * (x.size(-1) ** 0.5)
        uv = uv * suv
        u, v = torch.chunk(uv, 2, dim=-1)
        out = u * F.silu(v)
        out = self.c_proj(out)
        return self.dropout(out)


class NBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.attn = NSingleHeadCausalSelfAttention(n_embd)
        self.mlp = SwiGLUMLP(n_embd)

        # learnable residual interpolation rates
        self.base_scale = n_embd**-0.5
        self.attn_alpha = nn.Parameter(self.base_scale * torch.ones(n_embd))
        self.mlp_alpha = nn.Parameter(self.base_scale * torch.ones(n_embd))

    def forward(self, x):
        attn_out = self.attn(x)
        attn_alpha = torch.abs(self.attn_alpha).view(1, 1, -1)
        x = normalized_lerp(x, attn_out, attn_alpha)

        mlp_out = self.mlp(x)
        mlp_alpha = torch.abs(self.mlp_alpha).view(1, 1, -1)
        x = normalized_lerp(x, mlp_out, mlp_alpha)
        return x


class SingleHeadnGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[NBlock(n_embd) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Keep output scaling learnable (nGPT-style logits scaling)
        self.base_scale = n_embd**-0.5
        self.sz = nn.Parameter(self.base_scale * torch.ones(vocab_size))

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = l2_normalize(self.drop(x))
        x = self.blocks(x)
        x = l2_normalize(x)

        logits = self.lm_head(x)
        logits = logits * (torch.abs(self.sz).view(1, 1, -1) / self.base_scale)

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


def main():
    model = SingleHeadnGPT().to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.3f}M parameters")

    # nGPT usually does not need weight decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
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


if __name__ == "__main__":
    main()

'''
1.079M parameters
step 0: train loss 4.1733, val loss 4.1744
step 200: train loss 3.1209, val loss 3.1432
step 400: train loss 2.5702, val loss 2.5634
step 600: train loss 2.3024, val loss 2.3156
step 800: train loss 2.1244, val loss 2.1470
step 1000: train loss 1.9966, val loss 2.0430
step 1200: train loss 1.9056, val loss 1.9877
step 1400: train loss 1.8270, val loss 1.9326
step 1600: train loss 1.7711, val loss 1.9073
step 1800: train loss 1.7233, val loss 1.8656
step 1999: train loss 1.6868, val loss 1.8416

First not, in the goxountempt.

BULET:
We suBET; fair thee would. Countime, I pray his wellain! saySed!

Xo very sovenges fearer, so worrow
Hemmost up!

POMETER:

LEONUS:
Whicy thou honquitted well your kings, my more trues, ring counsuik ful.

GLOUCESTO:
Withs post this she stay in on, ancless do nept;
When, hein yet as blood tyreed, and that would I best in me by with the poor maid;
But theer fell, and thoughn brother wus hand with take your dispeegce:
I who the worm dispeybettle op's an is bear?

SARest CAFmust Glancfull give buy your slomenting.
But would no my lord?

HASW:
Slain tyroesvin
'''