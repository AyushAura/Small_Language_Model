import os
import torch
import numpy as np
from model import GPT, GPTConfig

# Configuration settings optimized for laptop execution
batch_size = 16
block_size = 128
max_iters = 5000
eval_iters = 200
learning_rate = 3e-4
checkpoint_path = "checkpoint.pt"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Helper to fetch data batches from binary files
def get_batch(split):
    data = np.memmap('train.bin' if split == 'train' else 'validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

config = GPTConfig(block_size=block_size)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Track starting iteration
start_iter = 0

# Check if a saved checkpoint exists to resume from
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter'] + 1
    print(f"Resumed training from step {start_iter}")
else:
    print("No checkpoint found. Starting training from scratch...")

print("Starting training loop...")
model.train()
try:
    for iter in range(start_iter, max_iters):
        xb, yb = get_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Step {iter}/{max_iters}: Train Loss {loss.item():.4f}")
            
            # Save a checkpoint every 100 steps
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)

except KeyboardInterrupt:
    print("\nTraining paused safely by user! Checkpoint saved.")

# Save final parameters when completely finished
torch.save(model.state_dict(), "best_model_params.pt")
print("Training finished and final model saved to best_model_params.pt!")