# extraction_pipeline.py

# Install these manually if needed:
# pip install torch transformers pillow tqdm

import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from tqdm import tqdm

# Paths
dataset_path = "data/"  # Path to your images folder
output_path = "outputs/"
os.makedirs(output_path, exist_ok=True)

# Load BLIP-2 Model
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Process each image
for filename in tqdm(os.listdir(dataset_path)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Load image
            img_path = os.path.join(dataset_path, filename)
            raw_image = Image.open(img_path).convert('RGB')

            # Prompt to guide the model
            prompt = (
                "You are a medical assistant. From the prescription image, extract:\n"
                "- Patient Name\n- Doctor Name\n- List of Medicines (with Dosage, Frequency, Duration)\n"
                "- Special Instructions\nReturn the output in JSON format."
            )

            # Preprocessing
            inputs = processor(raw_image, text=prompt, return_tensors="pt").to('cuda')

            # Generate Output
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Save Output
            file_name = filename.split('.')[0] + ".json"
            with open(os.path.join(output_path, file_name), "w") as f:
                f.write(generated_text)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\n✅ All images processed. Outputs saved in 'outputs/' folder.")
