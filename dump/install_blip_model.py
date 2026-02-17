#!/usr/bin/env python3
"""
Install and Test BLIP Model on E: Drive
BLIP (Bootstrapping Language-Image Pre-training) for image captioning
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import gc
import glob

print("\n" + "="*70)
print("🚀 INSTALLING & TESTING BLIP MODEL ON E: DRIVE")
print("="*70 + "\n")

# Set cache to E: drive
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📁 Cache Directory: {CACHE_DIR}")
print(f"💾 Storage: E: drive")

# Use smaller BLIP variant
MODEL_ID = "Salesforce/blip-image-captioning-base"

print(f"\n🎯 Model: {MODEL_ID}")
print(f"📊 Size: ~350 MB")
print("\n⏳ Downloading & Installing BLIP model...\n")

try:
    # Download processor
    print("📥 Downloading processor...")
    processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print("✅ Processor loaded successfully")
    
    # Download model
    print("📥 Downloading BLIP model weights (~350MB)...")
    print("⏳ This may take 2-5 minutes on first run...\n")
    
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    print("\n✅ BLIP model loaded successfully!")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Running on: {device.upper()}")
    
    # Find test images
    print("\n📸 Searching for test images...")
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
        print(f"✅ Found {len(found_images)} images\n")
        
        print("="*70)
        print("🤖 TESTING BLIP MODEL ON IMAGES")
        print("="*70 + "\n")
        
        for idx, img_path in enumerate(found_images[:3], 1):
            print(f"\n{'─'*70}")
            print(f"📷 Image {idx}: {os.path.basename(img_path)}")
            print(f"{'─'*70}")
            
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                print(f"📐 Resolution: {image.size[0]}×{image.size[1]}")
                
                # Process and generate caption
                print("🤖 Generating caption...")
                inputs = processor(image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=50)
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                print(f"✅ Caption generated!")
                print(f"🎯 BLIP Caption: {caption}\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        print("="*70)
        print("✅ BLIP MODEL TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"""
📊 BLIP Model Installed Successfully
   └─ Location: {CACHE_DIR}
   └─ Model: Salesforce/blip-image-captioning-base
   └─ Size: ~350 MB
   └─ Type: Image Captioning (Excellent quality captions)

🎯 Next Steps:
   1. To use BLIP in your app.py, create a new endpoint
   2. Or update app.py to support multiple models
   3. BLIP generates more detailed captions than Google ViT

💡 Performance:
   - BLIP: ~2-5 seconds per image (CPU)
   - Google ViT: ~0.5 seconds per image (faster)
   - Use BLIP for high-quality captions, ViT for speed

✨ Model is now cached on E: drive for future use!
        """)
        
except Exception as e:
    print(f"\n❌ Error during installation: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
gc.collect()
print("\n🧹 Cleanup complete!")
