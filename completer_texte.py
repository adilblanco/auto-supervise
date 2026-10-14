"""
Auto-supervisé n°2 : COMPLÉTER UN TEXTE
=======================================

Idée : on lit un texte, et le réseau doit deviner la LETTRE SUIVANTE.

  texte :  "le chat dort"
  entrée : "le cha"   -> cible : "t"
  entrée : "e chat"   -> cible : " "
  entrée : " chat "   -> cible : "d"   ... etc.

Chaque position du texte donne un exemple d'entraînement GRATUIT :
la "bonne réponse" est simplement la lettre qui suit dans le texte.

C'est exactement le principe de GPT et des grands modèles de langage,
entraînés sur des milliards de mots sans aucune étiquette humaine
(eux prédisent des morceaux de mots plutôt que des lettres).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 1) Les données : du texte brut, rien d'autre
texte = """le chat dort sur le canapé. le chien dort dans le jardin.
le chat mange du poisson. le chien mange de la viande.
le chat joue avec la balle. le chien joue dans le jardin.
la petite fille lit un livre. le petit garçon lit un journal.
le soleil brille dans le ciel. la lune brille dans la nuit.
le chat dort dans le jardin. le chien dort sur le canapé.
la fille mange une pomme. le garçon mange du pain.
""" * 5

lettres = sorted(set(texte))                       # le "vocabulaire" : toutes les lettres
lettre_vers_id = {c: i for i, c in enumerate(lettres)}
id_vers_lettre = {i: c for c, i in lettre_vers_id.items()}
data = torch.tensor([lettre_vers_id[c] for c in texte])

# 2) Fabriquer les exemples automatiquement : 10 lettres -> la suivante
CONTEXTE = 10
X = torch.stack([data[i:i + CONTEXTE] for i in range(len(data) - CONTEXTE)])
Y = data[CONTEXTE:]                                # la lettre qui suit = l'étiquette
print(f"{len(X)} exemples créés à partir du texte, sans rien étiqueter.")
print(f"Exemple : entrée = '{''.join(id_vers_lettre[int(i)] for i in X[5])}'"
      f"  ->  cible = '{id_vers_lettre[int(Y[5])]}'\n")

# 3) Le modèle : embedding des lettres + réseau récurrent (GRU) + classifieur
class ModeleTexte(nn.Module):
    def __init__(self, nb_lettres):
        super().__init__()
        self.embedding = nn.Embedding(nb_lettres, 32)   # chaque lettre -> vecteur
        self.gru = nn.GRU(32, 128, batch_first=True)    # lit la séquence
        self.sortie = nn.Linear(128, nb_lettres)        # score pour chaque lettre

    def forward(self, x):
        h, _ = self.gru(self.embedding(x))
        return self.sortie(h[:, -1])                    # on prédit après la dernière lettre

model = ModeleTexte(len(lettres))

# 4) Entraînement : classification classique (cross-entropy sur les lettres)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(15):
    perm = torch.randperm(len(X))
    for i in range(0, len(X), 128):
        idx = perm[i:i + 128]
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 5 == 4:
        print(f"Époque {epoch + 1:2d} | perte = {loss.item():.3f}")

# 5) Prédiction : on donne un début de phrase, le modèle complète lettre par lettre
def completer(debut, nb_lettres=40):
    model.eval()
    resultat = debut
    with torch.no_grad():
        for _ in range(nb_lettres):
            x = torch.tensor([[lettre_vers_id[c] for c in resultat[-CONTEXTE:]]])
            lettre_suivante = model(x).argmax(1).item()   # la lettre la plus probable
            resultat += id_vers_lettre[lettre_suivante]
    return resultat

print()
for debut in ["le petit garçon ", "la lune bri", "le chien mange "]:
    print(f"Début : '{debut}'")
    print(f"  ->  '{completer(debut)}'\n")
