import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process a question with optional image input.")
parser.add_argument("--image_path", type=str, default=None, help="Path to the input image file (optional).")
parser.add_argument("--user_input", type=str, required=True, help="User question or input text.")
args = parser.parse_args()

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure Flash Attention 2 is used only if GPU is available
attn_implementation = "flash_attention_2" if DEVICE == "cuda" else "eager"

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,  # Efficient precision for newer GPUs
    _attn_implementation=attn_implementation,
)

# Move model to the GPU
model = model.to(DEVICE)

# Load the image if provided
image = None
if args.image_path:
    try:
        image = Image.open(args.image_path).convert("RGB")  # Ensure RGB format
    except FileNotFoundError:
        print(f"Error: The file '{args.image_path}' does not exist.")
        exit(1)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": args.user_input}],
    }
]

# If an image is provided, add it to the message content
if image:
    messages[0]["content"].insert(0, {"type": "image"})

# Preprocess
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image] if image else None, return_tensors="pt")

# Move inputs to the same device as the model
inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

# Generate response
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

# Output the result
print("Generated Description:")
print(generated_texts[0])

