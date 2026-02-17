#!/usr/bin/env python3
"""
Test ESP32 Orientation Data Sender
Simulates ESP32 sending calibrated orientation data to server
"""

import requests
import json
import time
import random

# Server endpoint
SERVER_URL = "http://localhost:5000"

# Sample orientation data based on your ESP32 output
orientation_samples = [
    {
        "direction": "NEUTRAL",
        "calibrated_ax": -0.083,
        "calibrated_ay": 0.092,
        "calibrated_az": 0.899,
        "confidence": 95.2
    },
    {
        "direction": "LEFT", 
        "calibrated_ax": -0.760,
        "calibrated_ay": 0.069,
        "calibrated_az": 0.485,
        "confidence": 87.5
    },
    {
        "direction": "RIGHT",
        "calibrated_ax": 1.034,
        "calibrated_ay": -0.166,
        "calibrated_az": 0.117,
        "confidence": 92.1
    },
    {
        "direction": "FORWARD",
        "calibrated_ax": -0.170,
        "calibrated_ay": -0.697,
        "calibrated_az": 0.609,
        "confidence": 89.8
    },
    {
        "direction": "BACK",
        "calibrated_ax": 0.130,
        "calibrated_ay": 0.933,
        "calibrated_az": 0.180,
        "confidence": 91.3
    },
    {
        "direction": "INVERTED",
        "calibrated_ax": 0.118,
        "calibrated_ay": 0.024,
        "calibrated_az": -1.088,
        "confidence": 96.7
    }
]

def send_orientation_data():
    """Send orientation data to server"""
    print("🧭 ESP32 Orientation Data Test")
    print("=" * 40)
    
    try:
        for i, sample in enumerate(orientation_samples, 1):
            # Add device ID and timestamp
            data = {
                **sample,
                "device_id": "ESP32_TEST_001",
                "timestamp": time.time()
            }
            
            print(f"📊 Sending {i}/{len(orientation_samples)}: {data['direction']}")
            print(f"   CAL_AX: {data['calibrated_ax']:.3f}, CAL_AY: {data['calibrated_ay']:.3f}, CAL_AZ: {data['calibrated_az']:.3f}")
            print(f"   Confidence: {data['confidence']:.1f}%")
            
            # Send to orientation endpoint
            response = requests.post(
                f"{SERVER_URL}/api/orientation-data",
                headers={'Content-Type': 'application/json'},
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"   ✅ Success: {response.json().get('message')}")
            else:
                print(f"   ❌ Failed: {response.text}")
            
            print("-" * 40)
            time.sleep(2)  # Wait 2 seconds between samples
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")

if __name__ == "__main__":
    send_orientation_data()