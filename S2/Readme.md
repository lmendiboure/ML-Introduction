# TP – Séance 2 
## Sur-apprentissage, sous-apprentissage et généralisation

### Objectifs
- Comprendre ce que signifient **underfitting** et **overfitting**.  
- Observer l’évolution des performances sur **train** et **test**.  
- Visualiser la **frontière de décision** pour différents modèles.  
- Introduire la **validation croisée** comme méthode d’évaluation plus robuste.  

🛠Plateforme : **Google Colab** ou Python 3 + scikit-learn.  

---

## Repères utiles (où chercher l'info)
- `make_moons` : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html  
- `cross_val_score` : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html  
- `KNeighborsClassifier` : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  

Astuce Colab : utilisez `NomClasse?` pour voir rapidement les paramètres disponibles.

---

## Étape 0 — Préparer l’environnement

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
```

**Q1.** Pourquoi garde-t-on `Pipeline` et `StandardScaler`, même avec seulement deux variables ?  
**Q2.** Vérifiez dans la doc de `make_moons` : que signifient les paramètres `n_samples` et `noise` ?  

---

## Étape 1 — Générer et visualiser les données

```python
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.title("Dataset make_moons")
plt.show()
```

**Q3.** Quelle est la taille de ce dataset ?  
**Q4.** Que se passerait-il si on mettait `noise=0` ? Et si on augmentait fortement `noise` ?  
*(Indice : essayez et observez la figure)*  

---

## Étape 2 — Séparer apprentissage et test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

**Q5.** Quelle proportion est réservée au test ?  
**Q6.** Pourquoi utilise-t-on encore `stratify=y` ici, alors que les classes semblent équilibrées ?  

---

## Étape 3 — Tester plusieurs valeurs de k

Complétez le code (remplacez les `...`) :

```python
k_values = range(1, 21)   # valeurs de k à tester
acc_train, acc_test = [], []

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=...))  # à compléter
    ])
    model.fit(..., ...)   # à compléter
    acc_train.append(model.score(X_train, y_train))
    acc_test.append(model.score(..., ...))  # à compléter

plt.plot(k_values, acc_train, label="train")
plt.plot(k_values, acc_test, label="test")
plt.xlabel("k")
plt.ylabel("accuracy")
plt.legend()
plt.show()
```

**Q7.** Pour quels k observe-t-on un **overfitting** (train ≫ test) ?  
**Q8.** Pour quels k observe-t-on un **underfitting** (train ≈ test mais faible) ?  
**Q9.** Quel k semble être un bon compromis ?  

---

## Étape 4 — Visualiser la frontière de décision

```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    plt.title(title)
    plt.show()

# Exemple avec k=1 et k=15
for k in [1, 15]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    model.fit(X_train, y_train)
    plot_decision_boundary(model, X, y, f"Frontière pour k={k}")
```

**Q10.** Décrivez la frontière obtenue avec k=1. Est-elle simple ou complexe ?  
**Q11.** Décrivez la frontière obtenue avec k=15. Est-elle simple ou complexe ?  
**Q12.** Quelle relation observez-vous entre la complexité de la frontière et le risque d’over/underfitting ?  

---

## Étape 5 — Validation croisée

```python
scores = cross_val_score(
    Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ]),
    X, y, cv=5
)
print("Scores de validation croisée:", scores)
print("Moyenne:", scores.mean())
```

**Q13.** Qu’est-ce que `cv=5` signifie dans cet exemple ?  
**Q14.** Pourquoi la validation croisée est-elle plus fiable qu’un simple train/test split ?  
**Q15.** Dans quels cas pensez-vous qu’elle est particulièrement utile ?  

---

## Étape 6 — Discussion et synthèse

**Q16.** Résumez avec vos mots la différence entre underfitting et overfitting.  
**Q17.** Si vous deviez conseiller un choix de k à un collègue, quelle démarche suivriez-vous (indices : courbes, CV) ?  
**Q18.** Quel est le lien entre ce TP et ce que vous avez fait avec le dataset Breast Cancer lors de la séance 1 ?  
**Q19.** (Recherche) Regardez la doc de `KNeighborsClassifier`. Quels autres paramètres que `n_neighbors` peuvent influencer le modèle ? Essayez-en un et observez l’effet.  

---

# Bilan attendu
À la fin de ce TP, vous devez être capables de :  
- Expliquer et reconnaître overfitting et underfitting.  
- Interpréter une courbe train/test en fonction d’un hyperparamètre.  
- Utiliser la validation croisée pour évaluer un modèle.  
- Comprendre que le choix des **hyperparamètres** est lié au compromis biais/variance.
