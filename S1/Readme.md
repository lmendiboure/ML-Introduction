# TP – Séance 1 (version active & guidée)  
## Premiers pas en apprentissage supervisé avec scikit-learn

### 🎯 Objectifs
- Explorer un **vrai dataset** (Breast Cancer) et comprendre sa structure (lignes, colonnes, classes).  
- Découvrir un algorithme simple : **k plus proches voisins (k-NN)**.  
- Évaluer un modèle : **accuracy** et **matrice de confusion**.  
- Comparer avec la **régression logistique**.  
- Tester et **justifier** des choix de paramètres (valeurs de *k*, normalisation, etc.).  

⏳ Durée cible : **~2h**  
🛠️ Plateforme : **Google Colab** (ou local avec Python 3 + scikit-learn).  

---

## 🧭 Repères utiles (où chercher l’info ?)
- Guide de démarrage : https://scikit-learn.org/stable/getting_started.html  
- `KNeighborsClassifier` : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
- `LogisticRegression` : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
- `Pipeline` : https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html  
- `StandardScaler` : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html  
- `train_test_split` : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html  
- `ConfusionMatrixDisplay` : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html  

Astuce Colab : vous pouvez afficher l’aide courte d’une classe avec `KNeighborsClassifier?` (ou `help(KNeighborsClassifier)`).

---

## Étape 0 — Préparer l’environnement
Créez un nouveau notebook et exécutez :

```python
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
```

**Q1.** À quoi servent `Pipeline` et `StandardScaler` dans un flux de travail ML ?  
👉 Indice : lisez les courts résumés des docs liées ci-dessus.

---

## Étape 1 — Explorer rapidement le dataset
```python
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

print("X shape:", X.shape)             # (n_samples, n_features)
print("y shape:", y.shape)
print("Features (10 premières):", feature_names[:10])
print("Classes:", target_names)

# Afficher 5 premières lignes sous forme de DataFrame pour mieux lire
pd.DataFrame(X, columns=feature_names).head()
```

**Q2.** Combien y a-t-il d’exemples et de variables ?  
**Q3.** Que représente **une ligne** de `X` dans la réalité (pas “une ligne de tableau” 😉) ?  
**Q4.** Que signifient les classes `target_names` (qui est 0 ? qui est 1) ?  
👉 Indice : ce dataset est médical (tumeur bénigne/maligne).

---

## Étape 2 — Séparer apprentissage et test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)
```

**Q5.** Quelle proportion est réservée au test ? Pourquoi ne pas tout utiliser pour l’entraînement **et** le test ?  
**Q6.** À quoi sert `stratify=y` ici ? (Astuce : garder la même proportion de classes.)

---

## Étape 3 — Construire un premier modèle k-NN (avec normalisation)
Complétez le pipeline (lisez la doc `Pipeline` et `KNeighborsClassifier`) :

```python
# TODO: complétez les '...'
knn = Pipeline(steps=[
    ('scaler', StandardScaler()),          # étape de mise à l'échelle
    ('clf', KNeighborsClassifier(n_neighbors=...))  # choisir k (entier > 0)
])

knn.fit(X_train, y_train)                  # apprentissage
y_pred_knn = knn.predict(X_test)           # prédictions
acc_knn = knn.score(X_test, y_test)        # accuracy
print(f"Accuracy k-NN (k=?): {acc_knn:.3f}")
```

**Q7.** Pourquoi **standardiser** les features avant k-NN ? (Il y a une histoire de **distance**…)  
**Q8.** Expliquez avec vos mots ce que font `fit()` et `predict()`.

---

## Étape 4 — Lire une matrice de confusion (sur vos vrais résultats)
```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
plt.show()
```

**Q9.** Sans chercher sur Internet : d’après vos sorties, que représentent les 4 cases (nommez-les) ?  
**Q10.** Quelle information apporte la matrice de confusion que l’accuracy ne montre pas ?

👉 Besoin d’un indice ? Voyez la doc `ConfusionMatrixDisplay` pour les libellés.

---

## Étape 5 — Choisir *k* de manière raisonnée (actif mais guidé)
Plutôt que de vous donner les valeurs, **proposez une liste de k impairs** allant de 1 à une borne raisonnable.  
*Règle pratique fréquente* : tester jusqu’à **≈ √(n_train)** (arrondi au supérieur), en ne prenant que des **impairs** (pour éviter les ex æquo en vote majoritaire).

```python
n_train = len(X_train)
k_max = int(np.ceil(np.sqrt(n_train)))    # borne haute recommandée
print("Taille train:", n_train, "→ k_max recommandé ≈", k_max)

# TODO : construisez une liste de k impairs entre 1 et k_max (inclus si impair)
# Par ex. [1, 3, 5, ...]
k_values = [ ... ]

print(k_values[:10], "...")
```

**Q11.** Quelle méthode avez-vous utilisée pour générer des valeurs **impaires** ? (boucle, slicing, `range` pas de 2, etc.)

---

## Étape 6 — Évaluer plusieurs k (boucle à compléter)
Complétez la boucle :

```python
results = []
for k in k_values:
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=...))   # TODO: utilisez k
    ])
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results.append((k, acc))

# Affichage des 10 premiers résultats pour contrôle
print(results[:10])
```

**Q12.** Quel est le **meilleur k** trouvé ? Quelle **accuracy** obtenez-vous ?  
Astuce Python : `best = max(results, key=lambda t: t[1])`

Optionnel (très conseillé) : tracer la courbe **k vs accuracy** avec `matplotlib` (une seule figure, pas de style spécifique demandé).

---

## Étape 7 — Comparer avec la régression logistique
Complétez puis évaluez :

```python
# TODO: pipeline StandardScaler + LogisticRegression
log_reg = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500))   # vous pouvez ajuster max_iter si warning
])
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
acc_log = log_reg.score(X_test, y_test)
print(f"Accuracy Logistique: {acc_log:.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log)
plt.show()
```

**Q13.** Lequel est meilleur sur ce dataset : votre **meilleur k-NN** ou la **logistique** ?  
**Q14.** En regardant les deux matrices de confusion, quels **types d’erreurs** diffèrent ? (FP vs FN par ex.)

👉 Pour comprendre `LogisticRegression` : voyez la doc et le paramètre `C` (régularisation).

---

## Étape 8 — Effet de la normalisation (vérification indispensable)
**Test de contrôle** : que se passe-t-il si vous **retirez le `StandardScaler`** pour k-NN ?  
(code quasi identique, mais sans l’étape `scaler`)

```python
knn_no_scaler = Pipeline(steps=[
    # ('scaler', StandardScaler()),   # (désactivé volontairement)
    ('clf', KNeighborsClassifier(n_neighbors=...))  # reprenez votre meilleur k
])
knn_no_scaler.fit(X_train, y_train)
print("Accuracy k-NN sans scaler:", knn_no_scaler.score(X_test, y_test))
```

**Q15.** Que concluez-vous sur l’importance de la **mise à l’échelle** pour k-NN ?  
👉 Si la différence est faible chez vous : essayez d’autres *k* et observez la matrice de confusion.

---

## Étape 9 — Mini-défi 🎯 (> 0.95 d’accuracy)
- Visez **> 0.95** d’accuracy : ajustez *k* ou des paramètres de la logistique (`C`, `penalty`).  
- Justifiez en **1–2 phrases** votre choix final.

**Q16.** Quel est votre **meilleur modèle** ? Paramètres et accuracy obtenue ?  
**Q17.** Quelle **limite** voyez-vous à n’utiliser que l’accuracy ? (indice : classes déséquilibrées, coût des erreurs…)

---

## Étape 10 — (Optionnel) Aller un peu plus loin
- Essayez `DecisionTreeClassifier` (doc : `sklearn.tree.DecisionTreeClassifier`).  
- Comparez rapidement l’accuracy et la matrice de confusion.

**Q18.** Observez-vous un **sur-ajustement** (accuracy train >> test) ? Comment le détecteriez-vous proprement à l’avenir ?
👉 Teaser S2 : **validation croisée** et **biais/variance**.

---

## ✅ À rendre
- Notebook propre **avec votre code complété**, vos **réponses Q1–Q18**, et vos figures (matrices de confusion, éventuelle courbe k vs accuracy).  
- Quelques phrases d’interprétation pour **justifier** vos choix.
