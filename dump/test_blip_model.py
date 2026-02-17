#!/usr/bin/env python3
"""
Test BLIP (Bootstrapping Language-Image Pre-training) Model
BLIP is excellent for image captioning and visual question answering
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
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

# BLIP model - excellent for image captioning
MODEL_ID = "Salesforce/blip-image-captioning-base"

print("🚀 Loading BLIP model for image captioning...")

try:
    processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print("✅ BLIP Processor loaded")
    
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    print("✅ BLIP model loaded successfully!")
    
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
        print(f"\n📸 Found {len(found_images)} images to test:")
        for idx, img_path in enumerate(found_images[:3], 1):  # Test first 3
            print(f"\n{'='*60}")
            print(f"🖼️  Image {idx}: {os.path.basename(img_path)}")
            print(f"{'='*60}")
            
            try:
                image = Image.open(img_path).convert("RGB")
                print(f"📐 Image size: {image.size}")
                
                # Prepare inputs
                inputs = processor(image, return_tensors="pt").to(device)
                
                # Generate caption
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=50)
                
                # Decode caption
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                print(f"🎯 BLIP Caption: {caption}")
                print()
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
        
        print(f"\n✅ BLIP model testing complete!")
        print(f"✨ Cache stored at: {CACHE_DIR}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
gc.collect()
print("🧹 Memory cleaned!")
