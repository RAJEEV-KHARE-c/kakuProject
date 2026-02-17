from PIL import Image
import glob

FRAME_FOLDER = "frames/*.png"
OUTPUT_GIF = "output.gif"
WIDTH = 64
HEIGHT = 32
DURATION = 100  # milliseconds per frame

frames = []

# Load and sort images
for file in sorted(glob.glob(FRAME_FOLDER)):
    img = Image.open(file).convert("RGBA")

    # Resize (OLED size)
    img = img.resize((WIDTH, HEIGHT))

    # Add black background if transparent
    black_bg = Image.new("RGBA", img.size, (0, 0, 0))
    final = Image.alpha_composite(black_bg, img)

    frames.append(final.convert("P"))  # convert for GIF

# Save GIF
frames[0].save(
    OUTPUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=DURATION,
    loop=0
)

print("GIF created successfully ✅")
