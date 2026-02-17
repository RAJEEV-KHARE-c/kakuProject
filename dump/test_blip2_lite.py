#!/usr/bin/env python3
"""
Test BLIP-2 Lite Model - Much faster and smaller than BLIP-base
"""

import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import os
import gc
import glob

# Free up memory
gc.collect()

# Set cache to E: drive
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📁 Cache location: {CACHE_DIR}")

# BLIP-2 Lite model - Much smaller and faster
MODEL_ID = "Salesforce/blip2-opt-2.7b"

print("🚀 Loading BLIP-2 model for image captioning...")

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print("✅ BLIP-2 Processor loaded")
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    print("✅ BLIP-2 model loaded successfully!")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"📱 Using device: {device}")
    
    # Find images to test
    image_patterns = [
        "uploads/images/*.jpg",
        "uploads/images/*.jpeg",
        "uploads/images/*.png",
    ]
    
    found_images = []
    for pattern in image_patterns:
        found_images.extend(glob.glob(pattern))
    
    if not found_images:
        print("❌ No images found to test")
        print("📁 Checked folder: uploads/images/")
    else:
        print(f"\n📸 Found {len(found_images)} images")
        
        for idx, img_path in enumerate(found_images[:2], 1):  # Test first 2
            print(f"\n{'='*60}")
            print(f"🖼️  Image {idx}: {os.path.basename(img_path)}")
            print(f"{'='*60}")
            
            try:
                image = Image.open(img_path).convert("RGB")
                print(f"📐 Image size: {image.size}")
                
                # Prepare inputs
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Generate caption
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_length=50)
                
                # Decode caption
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"🎯 BLIP-2 Caption: {caption}")
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
        
        print(f"\n✅ BLIP-2 model testing complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
gc.collect()
print("🧹 Memory cleaned!")
