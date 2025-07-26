import time 
import torch 
from torch import nn 
from torch.nn import functional as F 
from model import Config, MinGPT
from utils import save_training_results

config = Config()
# ------------------ Load data--------------------
with open("data/taylor_swift.txt", "r", encoding="utf-8") as f:
        data = f.read().lower()

words = data.split() 
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}

encode = lambda s: [word_to_idx[c] for c in s]               # encoder: take a string, output a list of integers
decode = lambda l: ' '.join([idx_to_word[i] for i in l])     # decoder: take a list of integers, output a string

# Convert the entire text into a list of indices
data = torch.tensor([word_to_idx[c] for c in words], dtype=torch.long) 

#  Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Manual batch generator
def get_batch(split,batch_size=32, block_size=32):
    data_src = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_src) - block_size, (batch_size,))

    X = torch.stack([data_src[i:i+block_size] for i in ix])
    y = torch.stack([data_src[i+1:i+1+block_size] for i in ix])

    return X,y

# -------------------------------------------------------------


model = MinGPT()
model.to(config.device)
optimizer = torch.optim.Adam(model.parameters(),lr=config.lr,weight_decay=1e-4)

train_losses = []
val_losses = []

print(f"Vocab size is {vocab_size}, number of words is {len(words)}")
start = time.time()
for epoch in range(config.max_epochs):

    xb, yb = get_batch('train')
    logits, train_loss = model(xb,yb)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())
   
    if epoch % config.eval_itervals == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            xb, yb = xb.to(config.device), yb.to(config.device)
            logits, val_loss = model(xb,yb)
            print(f"Epoch {epoch}, Train Loss: {train_loss.item()} | Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
             
end = time.time()
print(f"Training finished in {end - start:.2f} seconds")
print(f"Final train loss is {train_loss.item()}")
save_training_results(train_losses, val_losses)

# context = torch.zeros((1,1), dtype=torch.long, device=config.device)
# print(decode(model.generate(idx = context, max_new_tokens=500)[0].tolist()))