
# ====================================
# PRODUCCIÓN: Cargar CLIP desde .pth
# ====================================

import torch
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

# Rutas
models_dir = Path('./models')
clip_path = models_dir / 'clip_model.pth'

# Cargar processor (rápido, ~1 segundo)
clip_processor = CLIPProcessor.from_pretrained(
    str(models_dir / 'clip_processor')
)

# Cargar modelo desde .pth (rápido, ~5-10 segundos)
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

checkpoint = torch.load(clip_path, map_location='cpu')
clip_model.load_state_dict(checkpoint['model_state_dict'])
clip_model.eval()

print(f"✅ CLIP loaded from .pth in ~5-10 seconds")

# Usar para generar embeddings de nuevas imágenes
from PIL import Image
def encode_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()

# Ejemplo
embedding = encode_image("new_product.jpg")
print(f"Embedding shape: {embedding.shape}")  # (1, 512)
