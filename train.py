import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import tiktoken
import time
import numpy as np
import math
from model import GPT
from model import GPTConfig

def load_tokens(filename):
    print(f"\nLOADING: {filename}")
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    print(f"loaded {len(ptt):,} tokens")
    return ptt


class DataLoaderLite:

    def __init__(self, B, T, split):
        assert split in {"train", "val"}

        self.B = B
        self.T = T
        self.split = split
        split_dir = "train" if split == "train" else "test"
        data_root = os.path.join("NCERT", split_dir)

        shards = sorted(
            os.path.join(data_root, filename)
            for filename in os.listdir(data_root)
            if filename.endswith(".npy")
        )

        assert len(shards) == 1, (f"expected exactly one .npy file for NCERT/{split_dir}, "f"found {len(shards)}")

        self.filename = shards[0]
        self.tokens = load_tokens(self.filename)
        self.current_position = 0
        self.epoch = 0
        self.batch_size = B * T

        assert len(self.tokens) >= self.batch_size + 1, (f"{self.filename} contains only {len(self.tokens):,} tokens, "f"but a batch requires at least {self.batch_size + 1:,}")
        self.batches_per_epoch = ((len(self.tokens) - 1) // self.batch_size)
        assert self.batches_per_epoch > 0

        print(f"{split}: {len(self.tokens):,} tokens | "f"{self.batches_per_epoch:,} batches/epoch | "f"B={B}, T={T}")

    def reset(self):
        self.current_position = 0
        self.epoch = 0

    def next_batch(self):
        B, T = self.B, self.T
        if self.current_position + self.batch_size + 1 > len(self.tokens):
            self.current_position = 0
            self.epoch += 1

        buf = self.tokens[self.current_position:self.current_position + self.batch_size + 1]
        assert len(buf) == self.batch_size + 1
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += self.batch_size

        return x, y

    @property
    def epoch_progress(self):
        return self.current_position / len(self.tokens)

# -----------------------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")


print(device)
total_batch_size = 32768
B = 8 
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
tokens_per_step = total_batch_size
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
print(f"tokens per optimizer step: {tokens_per_step:,}")

train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

torch.set_float32_matmul_precision("high")

model_path = "model_19999_full.pt"
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
raw_model = model
print("loaded model:")
print(raw_model.config)

max_lr_adamw = 3e-4
min_lr_adamw = 3e-5

max_lr_muon = 1e-2
min_lr_muon = 1e-3

warmup_steps = 75
max_steps = 3050
def get_lr(it, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
muon_params = []
adamw_params = []

muon_names = {"attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight",}

for name, param in raw_model.named_parameters():
    if any(name.endswith(suffix) for suffix in muon_names):
        muon_params.append(param)
    else:
        adamw_params.append(param)

print(f"Muon parameters: {sum(p.numel() for p in muon_params):,}")
print(f"AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

muon_optimizer = torch.optim.Muon(muon_params,lr=max_lr_muon,momentum=0.95,weight_decay=0.01,adjust_lr_fn="original",)

adamw_optimizer = torch.optim.AdamW(adamw_params,lr=max_lr_adamw,betas=(0.9, 0.95),eps=1e-8,weight_decay=0.01,fused=(device_type == "cuda"),)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

use_compile = False

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step % 1000 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        if step > 0 and (step % 305 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            fullsave_path = os.path.join(log_dir, f"model_{step:05d}_full.pt")
            checkpoint = {
                "model": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
                "val_loss": val_loss_accum.item(),
                "tokens_seen": (step + 1) * total_batch_size,
                "epoch": train_loader.epoch,
            }
            torch.save(raw_model, fullsave_path)
            print(f"Saved model to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)

    if ((step > 0 and step % 305 == 0) or last_step) and not use_compile:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)

        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    model.train()
    muon_optimizer.zero_grad()
    adamw_optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr_adamw = get_lr(step, max_lr_adamw, min_lr_adamw)
    lr_muon = get_lr(step, max_lr_muon, min_lr_muon)

    for param_group in adamw_optimizer.param_groups:
        param_group["lr"] = lr_adamw

    for param_group in muon_optimizer.param_groups:
        param_group["lr"] = lr_muon

    muon_optimizer.step()
    adamw_optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    cumulative_tokens = (step + 1) * total_batch_size
    epoch_progress = train_loader.epoch_progress

    log_line = (
      f"step {step + 1:6d} | "
      f"epoch {train_loader.epoch:4d} | "
      f"epoch_progress {epoch_progress * 100:6.2f}% | "
      f"tokens {cumulative_tokens / 1e6:9.2f}M | "
      f"loss {loss_accum.item():.6f} | "
      f"lr_adamw {lr_adamw:.4e} | "
      f"lr_muon {lr_muon:.4e} |"
      f"norm {norm:.4f} | "
      f"tok/sec {tokens_per_sec:.0f} | "
      f"dt {dt * 1000:.1f}ms"
    )
    print(log_line)

    with open(log_file, "a") as f:
        f.write(
          f"{step + 1} train "
          f"{loss_accum.item():.6f} "
          f"epoch={train_loader.epoch:4d} "
          f"epoch_progress={epoch_progress:.6f} "
          f"tokens={cumulative_tokens} "
          f"lr_adamw {lr_adamw:.4e} | "
          f"lr_muon {lr_muon:.4e} |"
          f"norm={norm:.6f} "
          f"tok_sec={tokens_per_sec:.2f} "
          f"dt={dt:.6f}\n"
        )