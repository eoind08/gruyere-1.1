import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tiktoken
import os 
import multiprocessing as mp
from tqdm import tqdm

local_dir = "NCERT"
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

shard_size = int(2.5e7)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fields = ['Explanation']
df = pd.read_csv("NCERT_Dataset.csv", usecols=fields)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

def tokenize(text):
  tokens = enc.encode_ordinary(text)
  tokens.append(eot)
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
  tokens_np_uint16 = tokens_np.astype(np.uint16)
  return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def process_dataframe(df, split, pool):
    output_dir = os.path.join(DATA_CACHE_DIR, split)
    os.makedirs(output_dir, exist_ok=True)

    shard_index = 0
    all_tokens_np = np.empty((shard_size,),dtype=np.uint16)
    token_count = 0
    progress_bar = None
    explanations = df["Explanation"].tolist()

    for tokens in pool.imap(tokenize, explanations, chunksize=16):

        if token_count + len(tokens) < shard_size:

            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)

            if progress_bar is None:
                progress_bar = tqdm(total=shard_size,unit="tokens",desc=f"{split} shard {shard_index}")
            progress_bar.update(len(tokens))

        else:
            remainder = shard_size - token_count
            if progress_bar is not None:
                progress_bar.update(remainder)
                progress_bar.close()
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

            filename = os.path.join(output_dir,f"NCERT_{split}_{shard_index:06d}.npy")

            write_datafile(filename,all_tokens_np)
            print(f"Wrote {filename}")

            shard_index += 1

            leftover = len(tokens) - remainder

            all_tokens_np[:leftover] = tokens[remainder:]

            token_count = leftover

            progress_bar = None

    if token_count != 0:

        if progress_bar is not None:
            progress_bar.close()

        filename = os.path.join(output_dir,f"NCERT_{split}_{shard_index:06d}.npy")

        write_datafile(filename,all_tokens_np[:token_count])

        print(f"Wrote {filename}")


#--------------------------------------------------------------------------------------------

if __name__ == "__main__":

    cpu_count = os.cpu_count() or 1
    nprocs = max(1, cpu_count // 2)

    print(f"Using {nprocs} CPU processes")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    with mp.Pool(nprocs) as pool:
        print("\nProcessing training data...")
        process_dataframe(train_df,"train",pool)
        print("\nProcessing test data...")
        process_dataframe(test_df,"test",pool)
        
    print("\nFinished.")