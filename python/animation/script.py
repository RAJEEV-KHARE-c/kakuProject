from PIL import Image, ImageSequence
import glob
import os

# MODE 1: Convert frames to GIF first
FRAME_FOLDER = "frames/*.*"  # Accept any image format
FRAME_TO_GIF_OUTPUT = "output.gif"

# MODE 2: Convert GIF to header
INPUT_GIF = "input.gif"
OUTPUT_GIF = "output_clean.gif"
OUTPUT_HEADER = "mygif.h"

WIDTH = 64
HEIGHT = 32
DURATION = 100    # milliseconds per frame (for frame conversion)
THRESHOLD = 128   # brightness threshold

def create_gif_from_frames():
    """Convert image frames (any format) to GIF"""
    frames = []
    # Accept common image formats
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    frame_files = [f for f in sorted(glob.glob(FRAME_FOLDER)) if f.lower().endswith(image_exts)]
    if not frame_files:
        print(f"⚠️ No frames found in {FRAME_FOLDER}")
        return None
    print(f"📂 Found {len(frame_files)} frames:")
    for file in frame_files:
        print(f"   - {os.path.basename(file)}")
        img = Image.open(file).convert("RGBA")
        # Fit image into (WIDTH, HEIGHT) with aspect ratio preserved and black padding
        img_ratio = img.width / img.height
        target_ratio = WIDTH / HEIGHT
        if img_ratio > target_ratio:
            # Image is wider than target: fit width
            new_w = WIDTH
            new_h = round(WIDTH / img_ratio)
        else:
            # Image is taller than target: fit height
            new_h = HEIGHT
            new_w = round(HEIGHT * img_ratio)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        # Create black background
        black_bg = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
        # Paste resized image centered
        paste_x = (WIDTH - new_w) // 2
        paste_y = (HEIGHT - new_h) // 2
        black_bg.paste(img_resized, (paste_x, paste_y), img_resized)
        frames.append(black_bg.convert("P"))  # convert for GIF
    # Save GIF
    frames[0].save(
        INPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0
    )
    print(f"✅ GIF created: {INPUT_GIF}")
    return INPUT_GIF

def process_frame(frame):
    frame = frame.convert("RGBA")
    # FIT (aspect ratio preserved, black bars)
    img_ratio = frame.width / frame.height
    target_ratio = WIDTH / HEIGHT
    if img_ratio > target_ratio:
        new_w = WIDTH
        new_h = round(WIDTH / img_ratio)
    else:
        new_h = HEIGHT
        new_w = round(HEIGHT * img_ratio)
    img_fit = frame.resize((new_w, new_h), Image.LANCZOS)
    fit_bg = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
    fit_bg.paste(img_fit, ((WIDTH - new_w) // 2, (HEIGHT - new_h) // 2), img_fit)
    gray = fit_bg.convert("L")
    bw = gray.point(lambda x: 255 if x > THRESHOLD else 0, mode="1")
    return [(bw, 'fit')]

def frame_to_bytes(img):
    pixels = img.load()
    data = []

    for y in range(HEIGHT):
        for x in range(0, WIDTH, 8):
            byte = 0
            for bit in range(8):
                if pixels[x + bit, y] == 255:
                    byte |= (1 << (7 - bit))
            data.append(byte)

    return data

# Step 1: Check if we need to create GIF from frames
if not os.path.exists(INPUT_GIF):
    print("🔍 Input GIF not found. Checking for frames...")
    create_gif_from_frames()

if not os.path.exists(INPUT_GIF):
    print("❌ Error: No input.gif found and no frames to process!")
    print("Please either:")
    print("  1. Place input.gif in this folder, OR")
    print("  2. Add PNG frames (1.png, 2.png, etc.) in frames/ folder")
    exit(1)

# Step 2: Process GIF to header file
print(f"🎬 Processing {INPUT_GIF}...")
gif = Image.open(INPUT_GIF)


frames_processed = []
frames_bytes = []
delays = []
scaling_labels = []

for frame in ImageSequence.Iterator(gif):
    processed_list = process_frame(frame)
    for processed, label in processed_list:
        frames_processed.append(processed)
        frames_bytes.append(frame_to_bytes(processed))
        delays.append(2000)  # 2 seconds for each scaling
        scaling_labels.append(label)

# ✅ Save cleaned GIF
frames_processed[0].save(
    OUTPUT_GIF,
    save_all=True,
    append_images=frames_processed[1:],
    duration=delays,
    loop=0
)

# ✅ Generate .h file
frame_count = len(frames_bytes)
bytes_per_frame = int(WIDTH * HEIGHT / 8)

with open(OUTPUT_HEADER, "w") as f:
    f.write("#ifndef MYGIF_H\n#define MYGIF_H\n\n")
    f.write("#include <pgmspace.h>\n\n")

    f.write(f"#define MYGIF_FRAME_COUNT {frame_count}\n")
    f.write(f"#define MYGIF_WIDTH {WIDTH}\n")
    f.write(f"#define MYGIF_HEIGHT {HEIGHT}\n\n")

    f.write("const uint16_t mygif_delays[MYGIF_FRAME_COUNT] = {")
    f.write(", ".join(str(d) for d in delays))
    f.write("};\n\n")

    f.write(f"PROGMEM const uint8_t mygif[{frame_count}][{bytes_per_frame}] = {{\n")

    for idx, frame in enumerate(frames_bytes):
        f.write("  {\n    ")
        for i, byte in enumerate(frame):
            f.write(f"0x{byte:02X}, ")
            if (i + 1) % 16 == 0:
                f.write("\n    ")
        f.write(f"\n  }}, // {scaling_labels[idx]}\n")

    f.write("};\n\n#endif\n")

print("\n✅ Done!")
print("📦 Generated files:")
print(f"   - {OUTPUT_GIF} (cleaned black & white GIF)")
print(f"   - {OUTPUT_HEADER} (Arduino/ESP32 header file)")
print(f"\n📊 Stats:")
print(f"   - Frames: {frame_count}")
print(f"   - Size: {WIDTH}x{HEIGHT}")
print(f"   - Bytes per frame: {bytes_per_frame}")
