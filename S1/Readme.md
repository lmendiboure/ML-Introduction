# TP – Séance 1 

## Premiers pas en apprentissage supervisé avec scikit-learn

### Objectifs
- Explorer un **vrai dataset** (Breast Cancer) et comprendre sa structure (lignes, colonnes, classes).  
- Identifier le **type de tâche** et le **type d’apprentissage** correspondant.  
- Découvrir deux algorithmes supervisés : **k plus proches voisins (k-NN)** et **régression logistique**.  
- Évaluer un modèle avec l’**accuracy** et la **matrice de confusion**.  
- Tester et justifier des choix de paramètres (valeurs de *k*, normalisation, hyperparamètres).  

🛠️ Plateforme : **Google Colab** ou Python 3 + scikit-learn.  

---

## Repères utiles (où chercher l’info)
- Guide de démarrage : https://scikit-learn.org/stable/getting_started.html  
- `KNeighborsClassifier` : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html  
- `LogisticRegression` : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
- `Pipeline` : https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html  
- `StandardScaler` : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html  
- `train_test_split` : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html  
- `ConfusionMatrixDisplay` : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html  

Astuce Colab : affichez l’aide courte d’une classe avec `KNeighborsClassifier?`.

---

## Étape 0 — Préparer l’environnement
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

**Q1.** À quoi sert `Pipeline` dans un flux de travail ML ?  
**Q2.** À quoi sert `StandardScaler` ?  
**Q3.** Pourquoi a-t-on besoin à la fois de modules pour les données (`train_test_split`, `StandardScaler`) et pour les modèles (`KNeighborsClassifier`, `LogisticRegression`) ?  

---

## Étape 1 — Explorer le dataset
```python
data = load_breast_cancer()
X, y = data.data, data.target
feature_names, target_names = data.feature_names, data.target_names

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Features (10 premières):", feature_names[:10])
print("Classes:", target_names)

pd.DataFrame(X, columns=feature_names).head()
```

**Q4.** Combien y a-t-il d’exemples (lignes) et de variables (colonnes) ?  
**Q5.** Que représente une **ligne** dans `X` ?  
**Q6.** Que signifient les classes `target_names` ? Qui est 0, qui est 1 ?  
**Q7.** Est-ce un problème de **classification** ou de **régression** ? Justifiez votre réponse.  

---

## Étape 2 — Séparer apprentissage et test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)
```

**Q8.** Quelle proportion est réservée au test ?  
**Q9.** Pourquoi ne pas tout utiliser pour l’apprentissage **et** le test ?  
**Q10.** À quoi sert `stratify=y` ?  

---

## Étape 3 — Construire un premier modèle k-NN (avec normalisation)
Complétez :

```python
knn = Pipeline(steps=[
    ('scaler', StandardScaler()),          
    ('clf', KNeighborsClassifier(n_neighbors=...))  # choisir k
])

knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = knn.score(X_test, y_test)
print(f"Accuracy k-NN (k=?): {acc_knn:.3f}")
```

**Q11.** Pourquoi standardiser les features avant k-NN ?  
**Q12.** Expliquez avec vos mots ce que font `fit()` et `predict()`.  
**Q13.** D’après le nom “k plus proches voisins”, comment le modèle prend-il une décision ?  

---

## Étape 4 — Lire une matrice de confusion
```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
plt.show()
```

**Q14.** Que représentent les 4 cases de la matrice ? 
**Q15.** Pourquoi la matrice de confusion donne plus d’informations que l’accuracy seule ?  

---

## Étape 5 — Choisir k de manière raisonnée
```python
n_train = len(X_train)
k_max = int(np.ceil(np.sqrt(n_train)))
print("Taille train:", n_train, "→ k_max recommandé ≈", k_max)

# TODO : générez une liste de valeurs impaires entre 1 et k_max inclus
k_values = [ ... ]
```

**Q16.** Quelle méthode utilisez-vous pour obtenir uniquement des valeurs **impaires** ?  
**Q17.** Pourquoi tester plusieurs k est-il nécessaire ?  

---

## Étape 6 — Évaluer plusieurs k
Complétez :

```python
results = []
for k in k_values:
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=k))
    ])
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results.append((k, acc))

print(results[:10])
```

**Q18.** Quel est le **meilleur k** trouvé ? Quelle est son accuracy ?  
**Q19.** En comparant petits et grands k, que remarquez-vous (sensibilité au bruit vs généralisation) ?  

---

## Étape 7 — Comparer avec la régression logistique
```python
log_reg = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500))
])
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
acc_log = log_reg.score(X_test, y_test)
print(f"Accuracy Logistique: {acc_log:.3f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log)
plt.show()
```

**Q20.** Quelle est l’accuracy de la régression logistique ?  
**Q21.** Lequel est meilleur : votre meilleur k-NN ou la logistique ?  
**Q22.** Selon vous, la régression logistique renvoie-t-elle une probabilité ou directement une classe ? Justifiez.  

---

## Étape 8 — Effet de la normalisation
```python
knn_no_scaler = Pipeline(steps=[
    ('clf', KNeighborsClassifier(n_neighbors=...))  # reprenez votre meilleur k
])
knn_no_scaler.fit(X_train, y_train)
print("Accuracy k-NN sans scaler:", knn_no_scaler.score(X_test, y_test))
```

**Q23.** Quelle différence observez-vous en retirant la normalisation ?  
**Q24.** Que concluez-vous sur l’importance du `StandardScaler` ?  

---

## Étape 9 — Mini-défi (Maximiser l'accuracy)
- Ajustez *k* ou les paramètres de la logistique (`C`, `penalty`).  
- Visez le maximum d’accuracy.  

**Q25.** Quel est votre meilleur modèle ? Donnez ses paramètres et son accuracy.  
**Q26.** Comment appelle-t-on le processus consistant à ajuster de tels paramètres ?

---

## Étape 10 — Limites de l’accuracy et ouverture
**Q27.** Quelle est la limite de l’accuracy comme seule métrique (pensez aux classes déséquilibrées ou au coût des erreurs) ?  

---

## À rendre
- Votre notebook avec code complété.  
- Réponses aux questions Q1–Q27.  
- Figures (matrices, courbe k vs accuracy).

## Bonus 1 — Courbes d’apprentissage

Nous voulons voir comment l’accuracy évolue quand on augmente la taille du jeu d’entraînement.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=500),
    X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

import matplotlib.pyplot as plt
plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="test")
plt.legend()
plt.show()
```

**QB1**. Que remarquez-vous lorsque la taille du jeu de train augmente ?
**QB2.** Comment interpréter un écart entre courbe train et test ? (indice : underfit vs overfit).

## Bonus 2 — Autres métriques que l’accuracy

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))
```

**QB3.** Que signifient précision (precision), rappel (recall) et F1-score ?
**QB4.** Pourquoi l’accuracy seule peut être trompeuse si les classes sont très déséquilibrées ?

## Bonus 3 — Visualiser la frontière de décision (2 features)

Réduisons le dataset à 2 variables pour mieux visualiser.

```python
X2 = X[:, :2]   # prenons les 2 premières features
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=42)

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X2_train, y2_train)

# Code de visualisation à compléter (scatter + contourf)
```

**QB5.** Dessinez la frontière de décision pour k-NN avec 2 features.
**QB6.** Faites de même pour la logistique. Quelle différence visuelle observez-vous entre les deux modèles ?
