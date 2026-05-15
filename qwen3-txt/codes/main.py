import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_path = "/models/Qwen3-0.6B"

    if os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        model_path = "Qwen/Qwen3-0.6B"
        print(f"Downloading model from ModelScope: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    print("Model loaded successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True,
    )

    print(f"\nQwen3 Response:\n{response}")


if __name__ == "__main__":
    main()
