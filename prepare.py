import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

def process(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}

if __name__ == '__main__':
    # 1. Load the TinyStories dataset from HuggingFace
    print("Loading dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    # Tokenize dataset splits
    tokenized = ds.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=4, # Safe now with the main guard
    )

    # 3. Save Tokenized Data to Binary Files (*.bin) using numpy memmap
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16  # GPT-2 vocab size (~50257) fits nicely in uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print("Data preparation complete! train.bin and validation.bin created.")