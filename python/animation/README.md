# GIF to Arduino Header Converter

This script converts GIF animations into black & white frames and generates a C header file for Arduino/ESP32.

## Folder Structure
```
python/animation/
├── frames/           # Store individual PNG frames here (optional)
├── script.py         # Main conversion script
├── input.gif         # Place your GIF file here
├── output_clean.gif  # Generated cleaned GIF
└── mygif.h           # Generated C header file
```

## How to Use

1. **Place your GIF file** in this directory as `input.gif`

2. **Run the script:**
```powershell
&"E:/Rajeev/esp 32/esp 32/.venv/Scripts/python.exe" script.py
```

3. **Output files generated:**
   - `output_clean.gif` - Cleaned black & white GIF
   - `mygif.h` - C header file ready for Arduino/ESP32

## Customization

Edit these variables in `script.py`:
- `WIDTH = 64` - Frame width in pixels
- `HEIGHT = 32` - Frame height in pixels  
- `THRESHOLD = 128` - Brightness threshold (0-255)

## What it does

1. Resizes each frame to 64x32 pixels
2. Converts to grayscale
3. Applies black background
4. Thresholds to pure black & white (binary)
5. Generates header file with frame data in `PROGMEM` format
6. Includes frame delays for animation timing
