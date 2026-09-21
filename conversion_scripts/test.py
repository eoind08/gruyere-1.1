from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "."

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
)

text = "The capital of France is"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

print("Tokenizer IDs:")
print(inputs["input_ids"])

print("\nLogits shape:")
print(outputs.logits.shape)