# TP3 – Découverte des réseaux de neurones avec le MLPClassifier

## Objectifs
- Comprendre le fonctionnement d’un réseau de neurones multicouche (**MLP**).
- Manipuler les principaux **hyperparamètres** : nombre de neurones, couches, fonctions d’activation, itérations.
- Observer les effets d’**underfitting**, d’**overfitting** et de **mauvaise convergence**.
- Relier ces phénomènes au **compromis biais / variance**.
- Approfondir l’intuition de la **descente de gradient** à travers la courbe de perte.

Durée cible : **2h à 2h30**  
Plateforme recommandée : **Google Colab** (Python 3 + scikit-learn)

---

## Étape 0 — Préparation et exploration des données
Nous allons travailler sur le dataset **Digits** de Scikit-learn..

**Aide** : documentation officielle de `load_digits`  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print("Taille du dataset :", digits.data.shape)
print("Nombre de classes :", len(digits.target_names))

# Visualisation rapide de quelques images
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
plt.show()
```

**Q1.** Qu'est ce que le dataset digits ? Combien d’images contient le dataset ? Quelle est la taille de chaque image ?  

**Q2.** Pourquoi les images doivent-elles être **aplaties** (converties en 64 colonnes) avant d’être utilisées par un MLP ?  

**Q3.** Combien de classes différentes contient ce jeu de données ?

---

## Étape 1 — Créer et entraîner un premier réseau simple

**Aide** : documentation `MLPClassifier`  
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Nous allons créer un réseau de neurones **avec une seule couche cachée** contenant 30 neurones.

```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Division du dataset en train/test
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42, stratify=digits.target
)

# TODO : Complétez les paramètres du MLP
mlp = MLPClassifier(hidden_layer_sizes=(...), activation='...', max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# TODO : prédisez sur le jeu de test
y_pred = mlp.predict(...)

print("Accuracy (train):", mlp.score(..., ...))
print("Accuracy (test):", accuracy_score(..., ...))

ConfusionMatrixDisplay.from_estimator(mlp, X_test, y_test)
plt.show()
```

**Q4.** Quelle est la précision obtenue sur le jeu d’entraînement et de test ?  
**Q5.** Que remarquez-vous dans la matrice de confusion ? Y a-t-il des chiffres plus difficiles à reconnaître ?  
**Q6.** Le modèle semble-t-il sous-apprendre ou sur-apprendre ? Pourquoi ?

---

## Étape 2 — Influence du nombre de neurones et de couches

Nous allons tester plusieurs architectures :  
- Une seule couche cachée avec 20, 50 et 100 neurones.  
- Puis un réseau **à deux couches** avec 50 et 20 neurones.

**Indice** : regardez le paramètre `hidden_layer_sizes` dans la doc de `MLPClassifier`.

```python
hidden_layers = [(20,), (50,), (100,), (50,20)]
results = []

for hl in hidden_layers:
    # TODO : créez le modèle et entraînez-le
    model = MLPClassifier(hidden_layer_sizes=hl, activation='relu', max_iter=300, random_state=42)
    model.fit(..., ...)
    acc_train = model.score(..., ...)
    acc_test = model.score(..., ...)
    results.append((hl, acc_train, acc_test))
    print(f"{hl} -> train={acc_train:.3f}, test={acc_test:.3f}")
```

**Q7.** Quelle architecture donne les meilleurs résultats sur le test ?  
**Q8.** Pour les architectures avec une seule couche, que se passe-t-il quand on augmente le nombre de neurones ?  
**Q9.** Le réseau à deux couches fait-il toujours mieux ? Expliquez pourquoi.  
**Q10.** Que pouvez-vous en conclure sur le lien entre **taille du modèle** et **généralisation** ?

---

## Étape 3 — Influence de la fonction d’activation

Nous allons comparer trois fonctions d’activation : `relu`, `tanh` et `logistic`.

**Aide** : paramètres `activation` dans la doc de `MLPClassifier`.

```python
activations = ['relu', 'tanh', 'logistic']

# TODO : complétez la boucle pour entraîner et évaluer chaque activation
for act in activations:
    model = MLPClassifier(hidden_layer_sizes=(50,20), activation=..., max_iter=300, random_state=42)
    model.fit(..., ...)
    print(f"Activation={act} -> train={model.score(..., ...):.3f}, test={model.score(..., ...):.3f}")
```

**Q11.** Quelle fonction d’activation obtient les meilleurs résultats ?  
**Q12.** Qu’observez-vous en termes de temps d’entraînement et de convergence ?  
**Q13.** Pourquoi `ReLU` est-elle souvent privilégiée dans les réseaux modernes ?

---

## Étape 4 — Nombre d’itérations et descente de gradient

Le paramètre `max_iter` contrôle le nombre d’itérations de l’algorithme d’optimisation (descente de gradient).  
Un message d’avertissement “**Maximum iterations reached**” indique que le modèle n’a pas complètement convergé.

**Aide** : consultez `mlp.loss_curve_` pour tracer la courbe d’évolution de l’erreur.

```python
mlp_iter = MLPClassifier(hidden_layer_sizes=(50,20), activation='relu', max_iter=30, random_state=42)
mlp_iter.fit(X_train, y_train)

# TODO : tracez la courbe de la loss
plt.plot(...)
plt.title("Courbe de perte (loss) au fil des itérations")
plt.xlabel("Itérations")
plt.ylabel("Loss")
plt.show()
```

**Q14.** Que représente la “loss” sur cette courbe ?  
**Q15.** Que se passe-t-il si vous augmentez `max_iter` à 200 ou 500 ?  
**Q16.** Que peut-on relier ici à la notion de **descente de gradient** vue en cours ?  
**Q17.** Pourquoi un nombre d’itérations trop élevé peut-il conduire à un sur-apprentissage ?

---

## Étape 5 — Synthèse

**Q18.** Résumez ce que vous avez observé :  
- effet du nombre de neurones,  
- effet du nombre de couches,  
- effet de la fonction d’activation,  
- effet du nombre d’itérations.  

**Q19.** Comment ces observations s’inscrivent-elles dans le compromis **biais / variance** ?  
**Q20.** Quelle combinaison d’hyperparamètres vous semble la plus équilibrée pour ce dataset ?  

---

## Pour aller plus loin

1. **QB1. Tester un réseau plus profond** : ajoutez une troisième couche cachée, par exemple `(100, 50, 20)`.  
   Observez les effets sur la précision et le temps d’entraînement.
     
3. **QB2. Comparer avec un modèle plus simple** : entraînez une régression logistique sur le même dataset (`LogisticRegression`).  
   Comparez les performances avec celles du MLP.  
4. **QB3. Visualiser les erreurs** : affichez quelques chiffres mal classés pour comprendre les confusions du modèle.

```python
import numpy as np
# TODO : complétez le code pour afficher 4 erreurs de prédiction
misclassified = np.where(... != ...)[0][:4]
for idx in misclassified:
    plt.imshow(...[idx].reshape(8,8), cmap='gray')
    plt.title(f"Vrai: {...[idx]} / Prédit: {...[idx]}")
    plt.show()
```
