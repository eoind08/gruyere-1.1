import torch

from model import GPT
from configuration_gruyere import GruyereConfig
from modeling_gruyere import GruyereForCausalLM


CHECKPOINT_PATH = "../log/model_03049_full.pt"
OUTPUT_DIR = "./Gruyere-1.1-HF"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Loading checkpoint on {DEVICE}...")

old_model = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

old_model.eval()

config = GruyereConfig(
    block_size=old_model.config.block_size,
    vocab_size=old_model.config.vocab_size,
    n_layer=old_model.config.n_layer,
    n_head=old_model.config.n_head,
    n_embd=old_model.config.n_embd,
)

new_model = GruyereForCausalLM(config)
new_model.load_state_dict(old_model.state_dict(), strict=True)
new_model.to(DEVICE)
new_model.eval()

print("Weights loaded successfully.")

torch.manual_seed(42)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (1, 128),
    dtype=torch.long,
    device=DEVICE,
)

with torch.inference_mode():
    old_logits, _ = old_model(input_ids)
    new_logits = new_model(input_ids).logits

max_difference = (old_logits - new_logits).abs().max().item()

print(f"Maximum logit difference: {max_difference:.10f}")

if max_difference > 1e-5:
    raise RuntimeError(
        f"Model conversion failed: maximum logit difference "
        f"is {max_difference}"
    )

print("Logits match.")

new_model.save_pretrained(OUTPUT_DIR)

config.save_pretrained(OUTPUT_DIR)

print(f"Saved HF model to {OUTPUT_DIR}")