# Apprentissage auto-supervisé

Petits exemples PyTorch pour comprendre l'apprentissage auto-supervisé.

## L'idée en une phrase

> **L'auto-supervisé, c'est du supervisé où les étiquettes sont générées automatiquement à partir des données.**

En supervisé, un humain fournit l'étiquette `y` (« cette image est un 7 »).
En auto-supervisé, on **cache une partie de la donnée**, et cette partie cachée devient la cible `y`.
La boucle d'entraînement reste classique (même `loss.backward()`, même optimiseur) : seule l'origine de `y` change.

```python
# Supervisé
X, y = images, etiquettes_humaines

# Auto-supervisé (image)
X = cacher_le_bas(images)   # entrée : la donnée avec un trou
y = images                  # cible : la partie cachée, qu'on avait déjà

# Auto-supervisé (texte)
X = texte[i : i+10]         # entrée : le début
y = texte[i+10]             # cible : la lettre suivante, qu'on avait déjà
```

## Quand a lieu le « cacher puis retrouver » ?

> Le « cacher puis retrouver » a donc lieu **pendant l'entraînement** : c'est l'exercice qu'on donne au modèle pour qu'il apprenne. Une fois entraîné, il peut compléter de nouvelles données (comme GPT qui complète ton texte), ou on peut réutiliser ce qu'il a appris pour une autre tâche.

## Les exemples

| Fichier | Tâche auto-supervisée | Entrée → cible |
|---|---|---|
| `completer_image.py` | Compléter les pixels d'une image (principe des *Masked Autoencoders*) | image dont le bas est caché → image complète |
| `completer_texte.py` | Compléter un texte (principe de GPT) | 10 lettres → la lettre suivante |

Les images de chiffres viennent de scikit-learn (`load_digits`) : rien à télécharger.

## Installation (macOS M1, Python 3.12)

```bash
micro create -n auto-superviser python=3.12
micro activate auto-superviser
pip install -r requirements.txt
```

## Lancer les exemples

```bash
python completer_image.py        # génère completer_image.png
python completer_texte.py
```

Chaque script tourne en quelques secondes sur CPU, sans GPU.

## Où l'auto-supervisé s'applique bien

On l'utilise quand on a **beaucoup de données brutes mais peu d'étiquettes**, car étiqueter coûte cher :

- **Texte** : prédire le mot suivant (GPT) ou un mot caché (BERT).
- **Images** : reconstruire des zones masquées (MAE), coloriser des photos en noir et blanc.
- **Audio** : prédire des morceaux de signal cachés (wav2vec).
- **Vidéo, séries temporelles** : prédire l'image ou la valeur suivante.
- **Médecine, biologie** : pré-entraîner sur des radiographies ou des protéines non annotées.
