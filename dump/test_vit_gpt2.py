#!/usr/bin/env python3
"""
ViT-GPT2 Image Captioning Model
Fast, lightweight alternative to BLIP/LLaVA (~500MB)
"""

import torch
from transformers import ViTFeatureExtractor, GPT2Tokenizer, ViTGPT2LMHeadModel
from PIL import Image
import os
import glob

print("\n" + "="*70)
print("🚀 VIT-GPT2 IMAGE CAPTIONING")
print("="*70 + "\n")

# Cache on E: drive
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📁 Cache: {CACHE_DIR}")
print(f"📊 Model Size: ~500MB (Much smaller than BLIP/LLaVA!)\n")

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

try:
    print("📥 Loading ViT-GPT2 model...")
    
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        MODEL_ID, 
        cache_dir=CACHE_DIR
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR
    )
    model = ViTGPT2LMHeadModel.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print("✅ ViT-GPT2 loaded successfully!")
    print(f"📱 Device: {device.upper()}\n")
    
    # Find images
    images = glob.glob("uploads/images/*.jpg") + glob.glob("uploads/images/*.png")
    
    if not images:
        print("❌ No images found")
    else:
        print(f"📸 Found {len(images)} images\n")
        print("="*70)
        print("🎯 VIT-GPT2 CAPTION RESULTS")
        print("="*70 + "\n")
        
        for idx, img_path in enumerate(images, 1):
            try:
                print(f"📷 {idx}. {os.path.basename(img_path)}")
                
                # Load image
                image = Image.open(img_path).convert("RGB")
                
                # Extract features
                pixel_values = feature_extractor(
                    images=image,
                    return_tensors="pt"
                ).pixel_values
                pixel_values = pixel_values.to(device)
                
                # Generate caption
                with torch.no_grad():
                    output_ids = model.generate(
                        pixel_values,
                        max_length=50,
                        num_beams=4
                    )
                
                # Decode
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                print(f"   Caption: {caption}\n")
                
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
        
        print("="*70)
        print("✅ ViT-GPT2 Captioning Complete!")
        print("="*70)
        print(f"""
📊 ViT-GPT2 Specifications:
   • Size: ~500MB (vs BLIP 990MB, LLaVA 14GB)
   • Speed: Fast (good for CPU)
   • Quality: Good captions
   • Memory: Efficient
   • Cached: E: drive

🎯 Best For:
   ✅ Real-time ESP32 dashboard
   ✅ Limited bandwidth
   ✅ Fast inference
   ✅ Edge devices
        """)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
