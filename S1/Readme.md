# TP – Séance 1 : Premiers pas en Machine Learning

## 🎯 Objectifs
- Explorer un vrai dataset et comprendre sa structure.
- Découvrir un premier algorithme : **k plus proches voisins (k-NN)**.
- Évaluer un modèle avec **accuracy** et **matrice de confusion**.
- Comparer avec un autre modèle simple : **régression logistique**.
- Apprendre à tester et analyser différents paramètres.

⏳ Durée : environ **2h**  
📍 Outil : **Google Colab**

---

## Étape 0 – Préparer l’environnement
Copiez-collez le code ci-dessous dans un nouveau notebook Google Colab :

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
```

**Q1.** À votre avis, pourquoi importer autant de bibliothèques différentes ?  
👉 Astuce : cherchez rapidement chaque module sur [scikit-learn.org](https://scikit-learn.org/stable/).

---

## Étape 1 – Explorer le dataset
```python
data = load_breast_cancer()
X = data.data
y = data.target
```

- Affichez la **taille** de `X` (`.shape`).  
- Affichez les **noms des variables** (`data.feature_names`).  
- Affichez les **classes cibles** (`data.target_names`).  

**Q2.** Combien d’exemples (lignes) et combien de variables (colonnes) contient le dataset ?  
**Q3.** Que représente une **ligne** dans X ?  
**Q4.** Quelles sont les deux classes à prédire (y=0 et y=1) ?

---

## Étape 2 – Découper en apprentissage et test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

**Q5.** Quelle proportion de données est réservée au test ?  
**Q6.** Pourquoi est-il important de séparer apprentissage et test ?  
**Q7.** Que se passerait-il si on utilisait le même jeu pour apprendre et tester ?

---

## Étape 3 – Premier modèle : k-NN
```python
knn = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=5))
])
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
```

**Q8.** Pourquoi ajoute-t-on un `StandardScaler` avant le classifieur ?  
**Q9.** Expliquez avec vos mots la différence entre `.fit()` et `.predict()`.  
**Q10.** Que signifie `n_neighbors=5` ? Que changerait k=1 ou k=20 ?

---

## Étape 4 – Évaluer le modèle
```python
print("Accuracy k-NN:", accuracy_score(y_test, y_pred_knn))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn)
```

**Q11.** Quelle est l’accuracy obtenue ?  
**Q12.** Dans la matrice de confusion, que représentent les 4 cases ?  
**Q13.** Pourquoi la matrice de confusion donne plus d’informations que l’accuracy seule ?

---

## Étape 5 – Tester plusieurs valeurs de k
```python
scores = []
for k in [1, 3, 5, 7, 9, 11]:
    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=k))
    ])
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    scores.append((k, acc))
    print(f"k={k}, accuracy={acc:.3f}")
```

**Q14.** Quelle valeur de k donne la meilleure accuracy ?  
**Q15.** Expliquez pourquoi un k trop petit ou trop grand peut poser problème.  
**Q16.** Tracez un graphique (k vs accuracy). Qu’observez-vous ?

---

## Étape 6 – Comparer avec la régression logistique
```python
log_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500))
])
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Accuracy Logistique:", accuracy_score(y_test, y_pred_log))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log)
```

**Q17.** Quelle est l’accuracy de la régression logistique ?  
**Q18.** Entre k-NN et logistique, quel modèle fonctionne le mieux sur ce dataset ?

---

## Étape 7 – Mini-défi 🎯
- Essayez d’obtenir une **accuracy > 0.95**.  
- Pour cela, vous pouvez :  
  - modifier la valeur de k,  
  - changer les paramètres de `LogisticRegression` (par ex. `penalty`, `C`),  
  - ou explorer d’autres classifieurs (`DecisionTreeClassifier` par exemple).  

**Q19.** Quel est votre meilleur modèle ? Quelle est son accuracy ?  
**Q20.** Quelle serait la prochaine étape pour aller plus loin (indices : validation croisée, autres métriques, etc.) ?

---
