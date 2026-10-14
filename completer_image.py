"""
Auto-supervisé n°1 : COMPLÉTER LES PIXELS D'UNE IMAGE
=====================================================

Idée : on cache la moitié du bas de chaque image, et le réseau doit
la redessiner.

-> L'entrée  = l'image avec le bas caché
-> La cible  = l'image complète (qu'on possède déjà !)
Aucune étiquette humaine : la "réponse" est l'image elle-même.

C'est le principe des modèles de type MAE (Masked Autoencoder) et
de l'"inpainting" (réparer une photo abîmée).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

torch.manual_seed(0)

# 1) Les données : seulement les images (8x8 pixels), pas les étiquettes
images = torch.tensor(load_digits().images, dtype=torch.float32) / 16.0
images = images.unsqueeze(1)                       # (1797, 1, 8, 8)
train, test = images[:1500], images[1500:]

# 2) Fabriquer l'entrée et la cible automatiquement
def cacher_le_bas(x):
    x_cache = x.clone()
    x_cache[:, :, 4:, :] = 0                       # on met à zéro les 4 lignes du bas
    return x_cache

X_train = cacher_le_bas(train)                     # entrée : image incomplète
Y_train = train                                    # cible  : image complète

# 3) Le modèle : un petit CNN qui sort une image de même taille
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
    nn.Linear(256, 64), nn.Sigmoid(),              # 64 pixels entre 0 et 1
    nn.Unflatten(1, (1, 8, 8)),                    # -> (N, 1, 8, 8)
)

# 4) Entraînement : perte = différence entre pixels prédits et vrais pixels
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(30):
    perm = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 64):
        idx = perm[i:i + 64]
        loss = F.mse_loss(model(X_train[idx]), Y_train[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 5 == 4:
        print(f"Époque {epoch + 1:2d} | perte = {loss.item():.4f}")

# 5) Prédiction sur des images jamais vues
model.eval()
originales = test[:6]
cachees = cacher_le_bas(originales)
with torch.no_grad():
    completees = model(cachees)

fig, axes = plt.subplots(3, 6, figsize=(9, 5))
for j in range(6):
    for i, (img, titre) in enumerate([(cachees, "entrée (bas caché)"),
                                      (completees, "complétée par le modèle"),
                                      (originales, "vraie image")]):
        axes[i, j].imshow(img[j, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_title(titre, loc="left", fontsize=9)
plt.tight_layout()
plt.savefig("completer_image.png")
plt.show()
