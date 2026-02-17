#!/usr/bin/env python3
"""
Quick BLIP Caption Generator
Uses cached model from E: drive
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import glob

# Use E: drive cache
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

MODEL_ID = "Salesforce/blip-image-captioning-base"

print("🚀 Loading BLIP from E: drive cache...\n")

try:
    # Load from cache (won't download again)
    processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ BLIP loaded from E: drive")
    print(f"📱 Device: {device}\n")
    
    # Get images
    images = glob.glob("uploads/images/*.jpg") + glob.glob("uploads/images/*.png")
    
    if not images:
        print("❌ No images found")
    else:
        print(f"📊 Analyzing {len(images)} images...\n")
        print("="*60)
        
        for idx, img_path in enumerate(images, 1):
            try:
                img = Image.open(img_path).convert("RGB")
                
                # Get caption
                inputs = processor(img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=50)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                print(f"\n📷 {idx}. {os.path.basename(img_path)}")
                print(f"🎯 Caption: {caption}\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        print("="*60)
        print("✅ Done!")
        
except torch.cuda.OutOfMemoryError:
    print("⚠️ GPU out of memory, using CPU instead...")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
