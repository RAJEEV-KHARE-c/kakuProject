#!/usr/bin/env python3
"""
LLaVA (Large Language and Vision Assistant) Model
Advanced vision-language model for image understanding and VQA
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import glob

print("\n" + "="*70)
print("🚀 LLAVA - LARGE LANGUAGE AND VISION ASSISTANT")
print("="*70 + "\n")

# Use E: drive cache
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📁 Cache: {CACHE_DIR}")
print(f"💾 Store: E: drive\n")

# Use HuggingFace optimized LLaVA (smaller than full version)
MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print(f"🤖 Model: LLaVA 1.5 7B (HF optimized)")
print(f"📊 Size: ~14GB\n")

try:
    print("📥 Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print("✅ Processor loaded")
    
    print("📥 Loading LLaVA model... (This may take a few minutes)")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✅ LLaVA model loaded!\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model.to(device)
    print(f"📱 Running on: {device.upper()}\n")
    
    # Find images
    print("📸 Searching for images...")
    images = glob.glob("uploads/images/*.jpg") + glob.glob("uploads/images/*.png")
    
    if not images:
        print("❌ No images found")
    else:
        print(f"✅ Found {len(images)} images\n")
        print("="*70)
        print("🎯 LLAVA ANALYSIS RESULTS")
        print("="*70 + "\n")
        
        for idx, img_path in enumerate(images[:3], 1):  # Analyze first 3
            try:
                print(f"📷 Image {idx}: {os.path.basename(img_path)}")
                
                # Load image
                image = Image.open(img_path).convert("RGB")
                print(f"📐 Size: {image.size[0]}×{image.size[1]}")
                
                # Prepare inputs
                prompt = "Describe this image in detail."
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                
                # Move to device
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate description
                print("🤖 Generating description...")
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7
                    )
                
                # Decode
                description = processor.decode(output[0], skip_special_tokens=True)
                
                # Extract just the response (remove prompt)
                if "Describe this image" in description:
                    description = description.split("Describe this image in detail.")[-1].strip()
                
                print(f"📝 Description: {description}\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
        
        print("="*70)
        print("✅ LLaVA Analysis Complete!")
        print("="*70)
        print(f"""
🎯 LLaVA Features:
   • Vision-language understanding
   • Detailed image descriptions
   • Can answer questions about images
   • Better context understanding
   • Cached on E: drive

📊 Performance:
   • Speed: Medium (2-5 seconds per image)
   • Quality: Excellent (detailed descriptions)
   • Memory: High (requires 14GB)
   • Best for: Detailed image analysis
        """)
        
except torch.cuda.OutOfMemoryError:
    print("⚠️ GPU out of memory")
    print("💡 Try using CPU or smaller model")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
