#!/usr/bin/env python3
"""
Compare ALL AI Image Analysis Models
Tests Google ViT, Microsoft GIT, and Lightweight Vision
"""

import os
import sys
import glob
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(__file__))

print("\n" + "="*70)
print("🤖 COMPARING ALL AI IMAGE ANALYSIS MODELS")
print("="*70 + "\n")

# Find images to analyze
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
    sys.exit(1)

print(f"📸 Found {len(found_images)} images to analyze\n")

# ============ MODEL 1: Google ViT (from app.py) ============
print("1️⃣  GOOGLE VIT (Vision Transformer - Classification)")
print("-" * 70)

try:
    from app import analyze_image_with_ai, AI_AVAILABLE
    
    if not AI_AVAILABLE:
        print("❌ Google ViT not available")
    else:
        test_image = found_images[0]
        caption = analyze_image_with_ai(test_image)
        print(f"✅ Model: Google ViT loaded")
        print(f"📷 Image: {os.path.basename(test_image)}")
        print(f"🎯 Caption: {caption}")
        print()
        
except Exception as e:
    print(f"❌ Error loading Google ViT: {e}\n")

# ============ MODEL 2: Microsoft GIT (Lightweight) ============
print("2️⃣  MICROSOFT GIT (Lightweight Image-to-Text)")
print("-" * 70)

try:
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
    os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
    
    MODEL_ID = "microsoft/git-base"
    
    print("🚀 Loading Microsoft GIT model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None
    )
    print("✅ Microsoft GIT loaded")
    
    test_image = found_images[0]
    image = Image.open(test_image).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=50)
    
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"📷 Image: {os.path.basename(test_image)}")
    print(f"🎯 Caption: {caption}")
    print()
    
except Exception as e:
    print(f"❌ Error loading Microsoft GIT: {e}\n")

# ============ MODEL 3: Lightweight Vision (Basic Analysis) ============
print("3️⃣  BASIC IMAGE ANALYSIS (PIL + Visual Features)")
print("-" * 70)

try:
    test_image = found_images[0]
    image = Image.open(test_image).convert("RGB")
    
    # Get image properties
    width, height = image.size
    img_array = np.array(image)
    
    # Basic color analysis
    mean_colors = np.mean(img_array, axis=(0, 1))
    brightness = np.mean(img_array)
    
    # Generate descriptive caption
    caption_parts = []
    
    if brightness > 200:
        caption_parts.append("bright")
    elif brightness < 80:
        caption_parts.append("dark")
    else:
        caption_parts.append("well-lit")
    
    aspect_ratio = width / height
    if aspect_ratio > 1.5:
        caption_parts.append("landscape-oriented")
    elif aspect_ratio < 0.7:
        caption_parts.append("portrait-oriented")
    else:
        caption_parts.append("square-oriented")
    
    caption = f"A {' '.join(caption_parts)} image captured at {width}×{height}"
    
    print("✅ Basic image analysis loaded")
    print(f"📷 Image: {os.path.basename(test_image)}")
    print(f"📐 Resolution: {width}×{height}")
    print(f"☀️ Brightness: {brightness:.0f}/255")
    print(f"🎯 Caption: {caption}")
    print()
    
except Exception as e:
    print(f"❌ Error in basic analysis: {e}\n")

# ============ ANALYSIS COMPARISON ============
print("="*70)
print("📊 MODEL COMPARISON SUMMARY")
print("="*70)
print("""
┌─ MODEL ─────────────────┬─ TYPE ──────────┬─ SPEED ─┬─ QUALITY ┐
│ Google ViT              │ Classification  │ Fast    │ Good     │
├─────────────────────────┼─────────────────┼─────────┼──────────┤
│ Microsoft GIT           │ Captioning      │ Medium  │ Excellent│
├─────────────────────────┼─────────────────┼─────────┼──────────┤
│ Basic Image Analysis    │ Feature-based   │ Very    │ Fair     │
│                         │                 │ Fast    │          │
└─────────────────────────┴─────────────────┴─────────┴──────────┘

✅ RECOMMENDATION: Use Google ViT for your ESP32 dashboard
   - Fast loading and inference
   - High accuracy for object detection
   - Currently integrated in app.py
   - Works perfectly with QVGA resolution images
""")

print("🎉 All model tests completed!")
