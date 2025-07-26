import time 
import torch 
from torch import nn 
from torch.nn import functional as F 

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    block_size = 256
    max_epochs = 1000
    eval_iters = 200
    eval_itervals = 300 
    n_emb = 128
    n_heads = 8
    head_size = n_emb // n_heads
    dropout = 0.2
    lr = 2e-3

vocab_size = 3149

config = Config()
class Head(nn.Module):
    """ Single head of self attention"""
    def __init__(self,head_size):
        super().__init__()
        self.Q = nn.Linear(config.n_emb,head_size,bias=False,device=config.device)
        self.K = nn.Linear(config.n_emb,head_size,bias=False,device=config.device)
        self.V = nn.Linear(config.n_emb,head_size,bias=False,device=config.device)
        self.register_buffer('tril',torch.tril(torch.ones(config.block_size,config.block_size))) 
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,X):
        """ 
        This function computes the attention scores and returns the output
        Args:
            X: input tensor of shape (B, T, C)
        Returns:
            out: output tensor of shape (B, T, C)
        """
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
    """ Multiple heads of self attention in parallel"""
    def __init__(self,num_heads,head_size):
        """
        Args:
            num_heads: number of attention heads
            head_size: size of each attention head
        Attributes:
            heads: list of attention heads
            proj: linear layer to project the concatenated output of all heads
            dropout: dropout layer
        Returns:
            out: output tensor of shape (B, T, n_emb)"""
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_emb,config.n_emb)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,X):
        """ Forward pass of the multi-head attention
        Args:
            X: input tensor of shape (B, T, n_emb)
        Returns:
            out: output tensor of shape (B, T, n_emb)
        """
        out = torch.cat([h(X) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out 

class FeedForward(nn.Module):
    """ A simple feedforward network"""
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
    """ Transformer block: Multi-head self attention + Feedforward network
    Args:
        n_emb: embedding size
        n_heads: number of attention heads
    Attributes:
        sa: Multi-head self attention
        ffwd: Feedforward network
        ln1: Layer normalization for self attention
        ln2: Layer normalization for feedforward network
    Returns:
        X: output tensor of shape (B, T, n_emb)
    """
    def __init__(self,n_emb,n_heads):
        """ Initialize the transformer block
        Args:
            n_emb: embedding size
            n_heads: number of attention heads
        Attributes:
            sa: Multi-head self attention
            ffwd: Feedforward network
            ln1: Layer normalization for self attention
            ln2: Layer normalization for feedforward network
        Returns:
            X: output tensor of shape (B, T, n_emb)
        """
        super().__init__()
        head_size = n_emb // n_heads
        self.sa = MultiHeadAttention(n_heads,head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self,X):
        """ Forward pass of the transformer block
        Args:
            X: input tensor of shape (B, T, n_emb)
        Returns:
            X: output tensor of shape (B, T, n_emb)"""
        X = X + self.sa(self.ln1(X))
        X = X + self.ffwd(self.ln2(X))
        return X
    
class MinGPT(nn.Module):
    """ 
    Minimal GPT model 
    Args:
        vocab_size: size of the vocabulary
    Attributes:
        token_embedding_table: embedding layer for tokens
        position_embedding_table: embedding layer for positions
        blocks: sequential blocks of transformer layers
        lm_head: linear layer for language modeling
    Returns:
        logits: output logits of shape (B, T, vocab_size)
        loss: loss value if targets are provided, otherwise None"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,config.n_emb)
        self.position_embedding_table = nn.Embedding(config.block_size,config.n_emb)
        self.blocks = nn.Sequential(
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            Block(config.n_emb,config.n_heads),
            nn.LayerNorm(config.n_emb)
        )
        self.lm_head = nn.Linear(config.n_emb,vocab_size) 

    def forward(self,X,targets=None):
        """
        Forward pass of the model
        Args:
            X: input tensor of shape (B, T)
            targets: target tensor of shape (B, T) for computing loss
        Returns:
            logits: output tensor of shape (B, T, vocab_size)
            loss: loss value if targets are provided, otherwise None
        """
        token_emb = self.token_embedding_table(X)
        position_ids = torch.arange(X.size(1), device=X.device)[None, :]  # shape [1, T]
        position_emb = self.position_embedding_table(position_ids)
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
    
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50):
        """
        Generate text from the model
        Args:
            idx: input tensor of shape (B, T)
            max_new_tokens: maximum number of new tokens to generate
            temperature: temperature for sampling
            top_k: number of top tokens to consider for sampling
        Returns:
            idx: output tensor of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k_logits, _ = torch.topk(logits, top_k)
                mask = logits < top_k_logits[:, [-1]]
                logits[mask] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx