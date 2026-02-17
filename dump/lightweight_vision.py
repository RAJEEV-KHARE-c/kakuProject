import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import gc

# Free up memory
gc.collect()

# Set cache to E: drive
CACHE_DIR = r"E:\Rajeev\esp 32\esp 32\.cache\huggingface"
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📁 Cache location: {CACHE_DIR}")

# Using a smaller model that requires less memory
MODEL_ID = "microsoft/git-base"  # Much smaller model ~250MB

IMAGE_PATH = r"E:\Rajeev\esp 32\esp 32\uploads\images\download.jpg"

print("🚀 Loading lightweight GIT model...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print("✅ Processor loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    print("✅ Lightweight model loaded successfully!")
    
    # Load image
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"📸 Image loaded: {image.size}")
    except FileNotFoundError:
        print(f"⚠️ Creating test image...")
        image = Image.new('RGB', (224, 224), color='blue')
    
    # Generate caption
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )
        
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n🧠 AI Caption: {caption}")
    
    print(f"\n✅ Success! Model cached at: {CACHE_DIR}")

except Exception as e:
    print(f"❌ Error with GIT model: {e}")
    print("\n🔄 Trying even smaller model...")
    
    # Fallback to very light model
    try:
        from transformers import pipeline
        captioner = pipeline(
            "image-classification", 
            model="google/vit-base-patch16-224",
            cache_dir=CACHE_DIR
        )
        result = captioner(image)
        print(f"🔍 Image Classification: {result[0]['label']} ({result[0]['score']:.2f})")
        
    except Exception as e2:
        print(f"❌ Fallback also failed: {e2}")

# Clean up
gc.collect()
print("🧹 Memory cleaned!")