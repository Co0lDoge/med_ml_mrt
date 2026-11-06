import io
import os
import requests
import numpy as np
from PIL import Image, ImageDraw

BASE = "http://localhost:8080/api/v1"
r = requests.post(f"{BASE}/login", json={"email": "doctor_demo@hospital.test", "password": "Password123!"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Check latest completed MRI study
mri_id = "0a29104f-d796-48db-b605-0331be28b5d3"
images = sorted(requests.get(f"{BASE}/mri/{mri_id}/images", headers=headers).json(), key=lambda x: x["page"])
nodes = requests.get(f"{BASE}/mri/{mri_id}/nodes", headers=headers).json()
if not nodes:
    mri_id = "ba564ced-2307-4b6a-9080-51b02cdf4a9f"
    images = sorted(requests.get(f"{BASE}/mri/{mri_id}/images", headers=headers).json(), key=lambda x: x["page"])
    nodes = requests.get(f"{BASE}/mri/{mri_id}/nodes", headers=headers).json()

segments = []
for n in nodes:
    segs = requests.get(f"{BASE}/mri/nodes/{n['id']}/segments", headers=headers).json()
    if isinstance(segs, list):
        segments.extend(segs)

print(f"Verified study {mri_id}: {len(images)} slices, {len(nodes)} node(s), {len(segments)} segment(s)")

seg_by_img = {}
for s in segments:
    seg_by_img.setdefault(s["image_id"], []).append(s)

# Render a 2x3 comparison grid of representative slices (sliceIndex 2, 3, 5, 7, 9, 11)
target_indices = [2, 3, 5, 7, 9, 11]
tile_size = 384
grid_img = Image.new("RGB", (tile_size * 3, tile_size * 2), (0, 0, 0))

for idx_pos, idx in enumerate(target_indices):
    img_item = images[idx]
    segs = seg_by_img.get(img_item["id"], [])
    raw = requests.get(f"{BASE}/download/{mri_id}/{img_item['id']}", headers=headers).content
    pil_img = Image.open(io.BytesIO(raw))
    arr = np.array(pil_img, dtype=np.float32)
    max_val = arr.max() if arr.max() > 0 else 1.0
    norm_arr = np.clip((arr / max_val) * 255.0, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(np.stack([norm_arr, norm_arr, norm_arr], axis=-1), mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for s in segs:
        pts = [(pt["x"], pt["y"]) for pt in s["contor"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(255, 59, 48, 95), outline=(255, 59, 48, 255), width=3)

    composited = Image.alpha_composite(rgb, overlay).convert("RGB")
    # Draw slice label
    label_draw = ImageDraw.Draw(composited)
    status_str = f"Корональный {idx+1}/{len(images)} (Page {img_item['page']}) | mask={len(segs)>0}"
    label_draw.rectangle([8, 8, 360, 34], fill=(0, 0, 0))
    label_draw.text((14, 14), status_str, fill=(255, 80, 80) if segs else (180, 180, 180))

    tile = composited.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    col = idx_pos % 3
    row = idx_pos // 3
    grid_img.paste(tile, (col * tile_size, row * tile_size))

out_path = "/mnt/c/Users/GigaDoge/.gemini/antigravity/brain/37e2096f-5a7b-4742-90a0-e83253019df6/overlay_validation_grid.png"
grid_img.save(out_path)
print(f"Saved visual overlay validation grid to {out_path}")
