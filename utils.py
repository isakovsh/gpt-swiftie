import os 
import matplotlib.pyplot as plt

def save_training_results(train_losses,val_losses,output_dir="training_results"):
    
    os.makedirs(output_dir, exist_ok=True)

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        val_x = [i for i in range(0, len(train_losses), 300)][:len(val_losses)]
        plt.plot(val_x, val_losses, label='Val Loss', linestyle='--')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(save_path)
    print(f"📈 Loss plot saved to: {save_path}")
    plt.close()