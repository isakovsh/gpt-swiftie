import torch 
from torch import nn 
from torch.nn import functional as F 


#------------------ Data loading --------------------------------- 

with open("taylor_swift.txt", "r", encoding="utf-8") as f:
        data = f.read().lower()

words = data.split() 
vocab = sorted(list(set(data)))
vocab_size = len(vocab)
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}

encode = lambda s: [word_to_idx[c] for c in s]              # encoder: take a string, output a list of integers
decode = lambda l: ''.join([idx_to_word[i] for i in l])     # decoder: take a list of integers, output a string

# Convert the entire text into a list of indices
data = torch.tensor([word_to_idx[c] for c in data], dtype=torch.long) 


# 1. Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 2. Manual batch generator
def get_batch(split,batch_size=32, block_size=32):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))

    X = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+1+block_size] for i in ix])

    return X,y

# -------------------------------------------------------------------

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    block_size = 256
    max_epochs = 5000
    eval_iters = 200
    eval_itervals = 300 
    n_emb = 64
    n_heads = 8
    head_size = n_emb // n_heads
    dropout = 0.2
    lr = 2e-3


config = Config()

torch.manual_seed(10090)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, y = get_batch('train')
            logits, loss = model(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.Q = nn.Linear(config.n_emb,head_size,bias=False,device=config.device)
        self.K = nn.Linear(config.n_emb,head_size,bias=False,device=config.device)
        self.V = nn.Linear(config.n_emb,head_size,bias=False,device=config.device)
        self.register_buffer('tril',torch.tril(torch.ones(config.block_size,config.block_size))) 
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,X):
        B, T, C = X.shape
        Q = self.Q(X)
        K = self.K(X)
        # compute attention scores 
        wei = Q @ K.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = torch.softmax(wei,dim=-1) 
        wei = self.dropout(wei)
        V = self.V(X)
        out = wei @ V
        # print(f"Single head ouput shape is {out.shape}")
        return out 

class MultiHeadAttention(nn.Module):
    """ Multiple heads of attention"""
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_emb,config.n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,X):
        out = torch.cat([h(X) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out 

class FeedForward(nn.Module):
    def __init__(self,n_emb):
        super().__init__()
        self.nnets = nn.Sequential(
            nn.Linear(n_emb,4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb,n_emb),
            nn.Dropout(config.dropout)
        )
    def forward(self,X):
        return self.nnets(X)

class Block(nn.Module):

    def __init__(self,n_emb,n_heads):
        super().__init__()
        head_size = n_emb // n_heads
        self.sa = MultiHeadAttention(n_heads,head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self,X):
        X = X + self.sa(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))
        return X
    
class MinGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size,config.n_emb)
        self.position_embedding_table = nn.Embedding(vocab_size,config.n_emb)
        self.blocks = nn.Sequential(
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            nn.LayerNorm(config.n_emb)
        )
        self.lm_head = nn.Linear(config.n_emb,vocab_size) 

    def forward(self,X,targets=None):

        token_emb = self.token_embedding_table(X)
        position_emb = self.position_embedding_table(X)
        emb = token_emb + position_emb
        X = self.blocks(emb)
        logits = self.lm_head(X)

        if targets is None:
            loss = None 
        
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits , loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # [B, C]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1) # [B, 1]
            idx = torch.cat((idx,idx_next),dim=1)
        
        return idx

model = MinGPT()
 
optimizer = torch.optim.Adam(model.parameters(),lr=config.lr)


for epoch in range(config.max_epochs):
   
    if epoch % config.eval_itervals == 0:
        losses = estimate_loss()
        print(f"Step {epoch} Train loss is {losses['train']:.4f}, Validation loss is {losses['val']:.4f}")

    else:
        xb, yb = get_batch('train')
        logits, loss = model(xb,yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"Final loss is {loss.item()}")
context = torch.zeros((1,1), dtype=torch.long, device=config.device)
print(decode(model.generate(idx = context, max_new_tokens=500)[0].tolist()))