import torch
import tiktoken
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

config = GPTConfig()
model = GPT(config)

# Load from checkpoint.pt instead of best_model_params.pt
checkpoint_path = "checkpoint.pt"
print(f"Loading weights from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Extract just the model's learned weights from the checkpoint dictionary
model.load_state_dict(checkpoint['model_state_dict'])

model.to(device)
model.eval()

enc = tiktoken.get_encoding("gpt2")
start_text = "I am Ayush"
input_ids = torch.tensor([enc.encode(start_text)], dtype=torch.long, device=device)

print(f"\nPrompt: {start_text}\n--- Generated Text ---")
generated_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
decoded_output = enc.decode(generated_ids[0].tolist())
print(decoded_output)