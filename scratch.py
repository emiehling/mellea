"""
Manual decoding with KV cache using HuggingFace Transformers.

This script demonstrates how to:
1. Load a small model (DistilGPT-2)
2. Use the KV cache for efficient autoregressive decoding
3. Generate tokens one at a time manually
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load a small model
model_name = "distilgpt2"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps")
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")


def default_decode(prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text using the default HuggingFace generate method."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], attention_mask= inputs["attention_mask"], max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def manual_decode_with_kv_cache(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Generate text using manual decoding with KV cache."""

    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)

    # First forward pass - process the entire prompt
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    generated_ids = input_ids.tolist()[0]

    # Manual decoding loop
    for step in range(max_new_tokens):
        # Apply temperature scaling
        scaled_logits = next_token_logits / temperature

        # Sample from the distribution
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Append to generated sequence
        generated_ids.append(next_token_id.item())

        # Check for EOS token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Forward pass with KV cache - only process the new token
        # This is the key efficiency gain: we don't recompute attention
        # for all previous tokens, just use the cached key/values
        with torch.no_grad():
            outputs = model(
                next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        # Print progress
        new_token = tokenizer.decode([next_token_id.item()])
        #print(f"Step {step + 1}: Generated '{new_token}'")

    # Decode the full sequence
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


def beam_search_with_kv_cache(
    prompt: str,
    max_new_tokens: int = 50,
    num_beams: int = 5,
    temperature: float = 0.7,
) -> None:
    """Generate text using beam search with KV cache."""
    # This function is a placeholder to demonstrate how one might implement
    # beam search with KV cache. The actual implementation would be more complex
    # and is not provided here for brevity.

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    num_beams=num_beams,            # beam width
    num_return_sequences=5,
    early_stopping=True,
    use_cache=True,                 # use the Transformer KV caching
    return_dict_in_generate=True,
    output_scores=True
    )

    for i, beam in enumerate(outputs.sequences):
        generated_text = tokenizer.decode(beam, skip_special_tokens=True)
        print(f"Beam {i + 1}: {generated_text}")

if __name__ == "__main__":
    prompt = "The meaning of life is"
    print(f"Prompt: {prompt}\n")
    print("=" * 80)

    default_result = default_decode(prompt, max_new_tokens=50)
    print(f"Default generation...\n{default_result}")

    
    result = manual_decode_with_kv_cache(prompt, max_new_tokens=50)

    print("=" * 80)
    print(f"\nFull generated text (using KV Cache):\n{result}")

    print("=" * 80)
    print("\n Beam search results:\n")
    beam_search_with_kv_cache(prompt, max_new_tokens=50, num_beams=5)
    