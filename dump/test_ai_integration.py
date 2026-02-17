#!/usr/bin/env python3
"""
Test script for AI Image Analysis Integration
Tests the analyze_image_with_ai function with existing images
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import analyze_image_with_ai, AI_AVAILABLE
import glob

def test_ai_analysis():
    print("🧪 TESTING AI IMAGE ANALYSIS INTEGRATION")
    print("=" * 50)
    
    if not AI_AVAILABLE:
        print("❌ AI modules not available for testing")
        return
    
    # Find images in uploads/images folder
    image_patterns = [
        "uploads/images/*.jpg",
        "uploads/images/*.jpeg", 
        "uploads/images/*.png",
        "static/*.jpg",
        "static/*.jpeg",
        "static/*.png"
    ]
    
    found_images = []
    for pattern in image_patterns:
        found_images.extend(glob.glob(pattern))
    
    if not found_images:
        print("❌ No images found to test")
        print("📁 Checked folders: uploads/images/ and static/")
        return
    
    print(f"📸 Found {len(found_images)} images to test:")
    for img in found_images:
        print(f"   • {img}")
    
    print(f"\n🤖 Testing AI analysis on first image...")
    test_image = found_images[0]
    
    try:
        caption = analyze_image_with_ai(test_image)
        print(f"\n✅ AI Analysis Result:")
        print(f"📷 Image: {os.path.basename(test_image)}")
        print(f"🎯 Caption: {caption}")
        print(f"\n🎉 AI integration test PASSED!")
        
    except Exception as e:
        print(f"❌ AI analysis test FAILED: {e}")

if __name__ == "__main__":
    test_ai_analysis()