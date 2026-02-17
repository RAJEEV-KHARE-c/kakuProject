#!/usr/bin/env python3
"""
Test BLIP Model on ESP32 Images
Analyzes images in uploads/images folder using BLIP image captioning
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import glob

print("\n" + "="*70)
print("🤖 TESTING BLIP MODEL ON ESP32 IMAGES")
print("="*70 + "\n")

# Set cache to E: drive
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ID = "Salesforce/blip-image-captioning-base"

print("🚀 Loading BLIP model...\n")

try:
    # Load processor and model
    processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ BLIP model loaded successfully!")
    print(f"📱 Device: {device.upper()}\n")
    
    # Find images
    image_patterns = [
        "uploads/images/*.jpg",
        "uploads/images/*.jpeg",
        "uploads/images/*.png",
    ]
    
    found_images = []
    for pattern in image_patterns:
        found_images.extend(glob.glob(pattern))
    
    if not found_images:
        print("❌ No images found in uploads/images/")
    else:
        print(f"📸 Found {len(found_images)} images to analyze\n")
        print("="*70)
        print("🎯 BLIP IMAGE ANALYSIS RESULTS")
        print("="*70 + "\n")
        
        for idx, img_path in enumerate(found_images, 1):
            try:
                # Load and process image
                image = Image.open(img_path).convert("RGB")
                print(f"📷 Image {idx}: {os.path.basename(img_path)}")
                print(f"📐 Resolution: {image.size[0]}×{image.size[1]}")
                
                # Generate caption
                inputs = processor(image, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=50)
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                print(f"🎯 Caption: {caption}")
                print()
                
            except Exception as e:
                print(f"❌ Error analyzing {os.path.basename(img_path)}: {e}\n")
        
        print("="*70)
        print(f"✅ BLIP Analysis Complete! ({len(found_images)} images analyzed)")
        print("="*70)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
