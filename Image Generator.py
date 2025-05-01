import subprocess
import sys
import importlib
import os

# ========== DEPENDENCY CHECK & INSTALL ==========
required_packages = ["ultralytics", "diffusers", "torch", "transformers", "opencv-python", "Pillow", "numpy", "accelerate"]

def install_if_missing(package):
    try:
        importlib.import_module(package if package != "opencv-python" else "cv2")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in required_packages:
    install_if_missing(pkg)

# ========== IMPORTS ==========
import torch
from diffusers import DiffusionPipeline
from ultralytics import YOLO as yolo
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

# ========== SET WORKING DIRECTORY TO SCRIPT'S LOCATION ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ========== DEVICE SETUP ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")

torch.cuda.empty_cache()

# ========== LOAD PIPELINE ==========
pipe = DiffusionPipeline.from_pretrained("dataautogpt3/Proteus-v0.6", torch_dtype=torch_dtype)
pipe = pipe.to(device)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))  # Optional: disable safety checker

# ========== VARIABLES ==========
animal = "bear"
quantity = 5
xnum = "2"
img_dir = os.path.join(SCRIPT_DIR, f"{animal}_{xnum}_images")
os.makedirs(img_dir, exist_ok=True)

# ========== GENERATE IMAGES ==========


for i in range(quantity):
    print(f"Attempt {i}")
    negative_prompt = "color, overlapping objects, touching objects, more than 3 animals,realistic textures, shadows, details, background elements, unclear outlines"
    prompt = f"A clean black silhouette of a single {animal}, centered, on a plain white background. No shadows, no details, no field."
    image = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=12, num_inference_steps=50).images[0]
    image.save(os.path.join(img_dir, f"{animal}_{i}.png"))
    if device == "cuda":
        torch.cuda.empty_cache()

print("Images saved successfully.")

# ========== FUNCTIONS ==========
def get_detection_table(yolo_output, model, conf_thresh=0.85):
    out_table = [[], []]
    for detection in yolo_output.boxes:
        if detection.conf[0] < conf_thresh:
            continue
        class_name = model.names[int(detection.cls[0])]
        bbox = detection.xyxy[0].tolist()
        out_table[0].append(class_name)
        out_table[1].append(bbox)
    return out_table

def resize_width_and_pad_height(img, padding_size=10, final_size=28):
    img = Image.fromarray(img)
    img_padded = ImageOps.expand(img, (padding_size,) * 4, fill=(255, 255, 255))
    return img_padded.resize((final_size, final_size), Image.Resampling.LANCZOS)

def to_black_n_white(image):
    bw_pic = np.array(image)
    bw_pic = cv2.cvtColor(bw_pic, cv2.COLOR_RGB2BGR)
    height, weight, _ = bw_pic.shape
    for x in range(height):
        for y in range(weight):
            pixel = bw_pic[x, y]
            if np.linalg.norm(pixel - [255,255,255]) > np.linalg.norm(pixel - [0,0,0]):
                bw_pic[x, y] = [0,0,0]
            else:
                bw_pic[x, y] = [255,255,255]
    return bw_pic

def isolate_detections(image_path, yolo_output, model, conf_thresh=0.8):
    original_image = Image.open(image_path)
    detection_table = get_detection_table(yolo_output, model, conf_thresh)
    labels, boxes = detection_table[0], detection_table[1]
    isolated_images = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_image = original_image.crop((x1, y1, x2, y2))
        new_box_size = max(x2 - x1, y2 - y1)
        isolated_image = Image.new("RGB", (new_box_size, new_box_size), color=(255, 255, 255))
        cropped_image.thumbnail((new_box_size, new_box_size), Image.Resampling.LANCZOS)
        paste_x = (new_box_size - cropped_image.width) // 2
        paste_y = (new_box_size - cropped_image.height) // 2
        isolated_image.paste(cropped_image, (paste_x, paste_y))
        isolated_images.append(to_black_n_white(isolated_image))
    return isolated_images

def sharpen_image(img, radius=2, percent=150, threshold=3):
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def Standardize(img):
    img = np.array(img)
    height, weight, _ = img.shape
    for x in range(height):
        for y in range(weight):
            pixel = img[x,y]
            if np.linalg.norm(pixel - [0,0,0]) > np.linalg.norm(pixel - [255,255,255]):
                img[x,y] = [0,0,0]
            else:
                img[x,y] = [255,255,255]
    return img

# ========== YOLO DETECTION ==========
model11 = yolo("yolo11l.pt", verbose=False)
model8 = yolo("yolov8l.pt", verbose=False)

yolo_out11, every_path11 = [], []
for root, _, files in os.walk(img_dir):
    for file in files:
        file_path = os.path.join(root, file)
        every_path11.append(file_path)
        yolo_out11.append(model11(file_path, verbose=False))

yolo_out8, every_path8 = [], []
for root, _, files in os.walk(img_dir):
    for file in files:
        file_path = os.path.join(root, file)
        every_path8.append(file_path)
        yolo_out8.append(model8(file_path, verbose=False))

# ========== FINAL OUTPUT ==========
final_output = []
for image_index in range(len(yolo_out11)):
    isolated_images = isolate_detections(every_path11[image_index], yolo_out11[image_index][0], model11)
    for img in isolated_images:
        t = Standardize(sharpen_image(resize_width_and_pad_height(img)))
        final_output.append(t)

# ========== SAVE ==========
save_path = os.path.join(SCRIPT_DIR, f"{animal}_{xnum}_out")
os.makedirs(save_path, exist_ok=True)
for idx, image in enumerate(final_output):
    img = Image.fromarray(image)
    img.save(os.path.join(save_path, f'image_{idx}.png'))
