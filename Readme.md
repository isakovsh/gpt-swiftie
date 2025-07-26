# 🎤 GPT-Swiftie: Train a Mini-GPT on Taylor Swift Lyrics
![SwiftGPT Banner](https://github.com/isakovsh/gpt-swiftie/blob/master/images/image.png) 

**GPT-Swiftie** is a lightweight GPT-style language model trained on Taylor Swift lyrics. Built from scratch using PyTorch, this project is great for learning how transformers work under the hood — from embeddings to multi-head self-attention to autoregressive text generation.

> 🔥 Followed by [nanoGPT](https://github.com/karpathy/nanoGPT) — minimal, educational, and Swiftie-themed.

---
![Training Loss](https://github.com/isakovsh/gpt-swiftie/blob/master/images/loss_plot.png) 
## 🚀 Features

- 🧱 Transformer architecture from scratch (no HuggingFace)
- 🧠 Custom attention, layer norm, and MLP blocks
- 🎶 Trains on Taylor Swift lyrics
- 📝 Inference via greedy text generation
- 🧪 Easy to experiment, extend, and learn

---

## 🗂️ Project Structure
```
gpt-swiftie/
├── data/       # Taylor Swift lyrics dataset
├── outputs.txt # Generated outputs
├── train.py    # Training script
├── generate.py # Text generation
└── README.md
```

## Install
```
pip install torch numpy tiktoken torch tqdm
```
---
## Quick start
```
python train.py
```
---
## Generate Examples 
```
python generate.py
```
---
## ✨ Example Output
```
"any time you assume i'm fine, but what would you do if i (i) [chorus] 
break free and leave us in ruins? took this dagger in me and removed it gain the weight and removed it gain 
[verse 2] it if i'm high line once i've been callin' one ends for so small and our town, 
i'm dead i could let the weight and fallin' back to your passport and if you would've been saying "yes" will sink face 
```

## TODOS
- Reduce overfitting

- Use subword tokenizer (e.g., BPE) for better generalization

- Train on multiple artists for cross-style generation

## 🌟 Acknowledgements

- Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)

- Lyrics sourced via [LyricsGenius API](https://github.com/johnwmillr/LyricsGenius)
