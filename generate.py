from model import MinGPT
import torch

with open("data/taylor_swift.txt", "r", encoding="utf-8") as f:
        data = f.read().lower()

words = data.split() 
vocab = sorted(list(set(words)))
vocab_size = len(vocab)
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [word_to_idx[c] for c in s]            
decode = lambda l: ' '.join([idx_to_word[i] for i in l])


model = MinGPT()
model.load_state_dict(torch.load('model_weights/min_gpt.pth'))
model.eval()

def generate_text(max_new_tokens=100, temperature=0.7, top_k=50):

    with torch.no_grad():

        context = torch.zeros((1,1), dtype=torch.long)
        generated_text = decode(model.generate(idx = context, max_new_tokens=500)[0].tolist())
    
    return generated_text

# Example usage
if __name__ == "__main__":
    generated_text = generate_text(max_new_tokens=10000, temperature=0.7, top_k=50)
    # save the generated text to a file
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)
    print("Generated text saved to 'generated_text.txt'")
    # print(generated_text)
   
                                     