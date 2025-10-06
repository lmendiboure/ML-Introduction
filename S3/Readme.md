# TP3 – Découverte des réseaux de neurones avec le MLPClassifier

## Objectifs
- Comprendre le fonctionnement d’un réseau de neurones multicouche (**MLP**).
- Manipuler les principaux **hyperparamètres** : nombre de neurones, couches, fonctions d’activation, itérations.
- Observer les effets d’**underfitting**, d’**overfitting**, de **mauvaise convergence**.
- Relier ces phénomènes au **compromis biais / variance**.
- Découvrir les notions clés : **loss, convergence, solver, softmax, epochs/batchs, hyperparamètres**.
- Approfondir l’intuition de la **descente de gradient**.

Durée cible : **2h à 2h30**  
Plateforme recommandée : **Google Colab** (Python 3 + scikit-learn)

---

## Étape 0 — Préparation et exploration des données

Nous allons travailler sur le dataset **Digits** de Scikit-learn.

📚 **Aide :** [Documentation officielle – `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

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

**Q1.** Qu'est ce que ce dataset ? Combien d’images contient-il ? Quelle est la taille de chaque image ?  
**Q2.** Pourquoi les images doivent-elles être **aplaties** (converties en 64 colonnes) avant d’être utilisées par un MLP ?  
**Q3.** Combien de classes différentes contient ce jeu de données ?  

**Concept : `random_state`**
> Le paramètre `random_state` fixe la graine aléatoire pour rendre vos expériences **reproductibles**.  
> Si vous l’enlevez, les résultats peuvent légèrement varier d’une exécution à l’autre.  

**Q3bis.** Exécutez deux fois le même modèle avec et sans `random_state`. Que constatez-vous ?  

---

## Étape 1 — Créer et entraîner un premier réseau simple

📚 **Aide :** [Documentation – `MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

Nous allons créer un réseau de neurones **avec une seule couche cachée** contenant 30 neurones.

```python
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report

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

---

### Découverte du rapport de classification

Scikit-learn fournit un résumé des performances appelé **`classification_report`**, contenant plusieurs métriques :  

| Terme | Signification intuitive |
|-------|--------------------------|
| **precision** | parmi les exemples prédits dans une classe, combien étaient corrects ? |
| **recall (rappel)** | parmi les exemples réellement dans cette classe, combien ont été bien trouvés ? |
| **f1-score** | moyenne harmonique précision/rappel – équilibre entre les deux. |
| **support** | nombre d’exemples réels dans la classe. |

```python
from sklearn.metrics import classification_report
# TODO : affichez le rapport complet
print(classification_report(..., ...))
```

**Q5b.**  
1. Quelle différence voit-on entre **precision** et **recall** ?  
2. Pourquoi certaines classes ont-elles un **f1-score** plus bas que d’autres ?  
3. Que représente la colonne **support** ?  
4. Quelle métrique te semble la plus “juste” globalement ?  

📚 **Aide :** [Documentation – `classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

---

 **Sorties probabilistes – Fonction Softmax**
> Le réseau ne “devine” pas une seule classe : il calcule une **probabilité** pour chaque chiffre possible (0–9).  
> Ces probabilités sont normalisées par une fonction appelée **Softmax**, de sorte que leur somme = 1.

```python
# TODO : affichez les probabilités de prédiction pour 5 chiffres
probas = mlp.predict_proba(X_test[:5])
print(probas)
```

**Q5c.** Pourquoi la somme de chaque ligne (probabilités) vaut-elle toujours 1 ?  

---

## Étape 2 — Influence du nombre de neurones et de couches

Nous allons tester plusieurs architectures :  
- une seule couche cachée avec 20, 50 et 100 neurones ;  
- puis un réseau **à deux couches** avec 50 et 20 neurones.

**Indice :** regardez le paramètre `hidden_layer_sizes` dans la doc de `MLPClassifier`.

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
    print(f"{str(hl):>10} -> train={acc_train:.3f}, test={acc_test:.3f}")
```

**Q7.** Quelle architecture donne les meilleurs résultats sur le test ?  
**Q8.** Que se passe-t-il quand on augmente le nombre de neurones ?  
**Q9.** Le réseau à deux couches fait-il toujours mieux ? Expliquez.  
**Q10.** Quel lien voyez-vous entre **taille du modèle** et **généralisation** ?  

 **Concept – Hyperparamètres :**
> Les réglages choisis avant l’apprentissage (ex. nombre de couches, neurones, activation…) sont des **hyperparamètres**.  
> Le modèle apprend ensuite les **poids et biais** internes.  
> Plus tard, nous verrons comment automatiser le choix des hyperparamètres via la **validation croisée**.

---

## Étape 3 — Influence de la fonction d’activation

Nous allons comparer trois fonctions d’activation : `relu`, `tanh` et `logistic`.

**Aide :** [Documentation – `MLPClassifier.activation`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

```python
activations = ['relu', 'tanh', 'logistic']

# TODO : complétez la boucle pour entraîner et évaluer chaque activation
for act in activations:
    model = MLPClassifier(hidden_layer_sizes=(50,20), activation=..., max_iter=300, random_state=42)
    model.fit(..., ...)
    print(f"Activation={act} -> train={model.score(..., ...):.3f}, test={model.score(..., ...):.3f}")
```

**Q11.** Quelle fonction d’activation obtient les meilleurs résultats ?  
**Q12.** Qu’observez-vous sur la vitesse et la stabilité d’apprentissage ?  
**Q13.** Pourquoi `ReLU` est-elle souvent privilégiée dans les réseaux modernes ?  

---

## Étape 4 — Nombre d’itérations, convergence et descente de gradient

**Aide :** [Doc complète – Neural networks: training](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

Le paramètre `max_iter` contrôle le nombre d’**itérations** du processus d’apprentissage (descente de gradient).  
Un message *"Maximum iterations reached and the optimization hasn't converged yet"* signifie que le modèle n’a pas complètement convergé.

**Concept – Convergence :**  
> Un modèle “converge” lorsque la **loss (erreur)** ne diminue plus significativement.  
> Sinon, il faut plus d’itérations (`max_iter`) ou ajuster le **solver** (algorithme d’optimisation).

**Concept – Solver :**  
> Le paramètre `solver` choisit la méthode d’optimisation :  
> - `'adam'` (par défaut) : rapide et robuste.  
> - `'sgd'` : descente de gradient stochastique (plus bruitée).  
> - `'lbfgs'` : plus précis mais lent.  

**Concept – Fonction de coût (loss)**  
> La loss mesure l’erreur moyenne entre les prédictions et les vraies classes.  
> Le réseau cherche à la **minimiser**.  
> Par défaut, scikit-learn utilise la **log-loss (entropie croisée)**.  

**Concept – Epoch et batch :**  
> - Une **epoch** = un passage complet sur les données d’entraînement.  
> - L’apprentissage est souvent fait par “batches” (sous-parties du dataset).  
> - Chaque batch met à jour les poids via la descente de gradient.  

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
**Q14bis.** Pourquoi la loss ne descend-elle jamais sous zéro ? Quelle relation avec la précision (accuracy) ?  
**Q15.** Que se passe-t-il si vous augmentez `max_iter` à 200 ou 500 ?  
**Q16.** Comment relier cela à la **descente de gradient** vue en cours ?  
**Q16bis.** Que veut dire “le modèle n’a pas convergé” ? Comment le faire converger ?  
**Q16ter.** Essayez de changer `solver='sgd'` – que remarquez-vous ?  
**Q17.** Pourquoi trop d’itérations peuvent-elles conduire à un sur-apprentissage ?  
**Q17bis.** Si votre modèle converge mal, que pourriez-vous changer : `max_iter` ou la taille des batches ? Pourquoi ?  

---

## Étape 5 — Synthèse et réflexion

**Q18.** Résumez ce que vous avez observé :  
- effet du nombre de neurones,  
- effet du nombre de couches,  
- effet de la fonction d’activation,  
- effet du nombre d’itérations.  

**Q19.** Comment ces observations s’inscrivent-elles dans le compromis **biais / variance** ?  
**Q20.** Quelle combinaison d’hyperparamètres vous semble la plus équilibrée pour ce dataset ?  
**Q20bis.** Parmi tous les paramètres du MLP, lesquels sont **appris** et lesquels sont **hyperparamètres** ?  

---

## Pour aller plus loin (optionnel)

1. **Tester un réseau plus profond :** ajoutez une 3e couche cachée `(100,50,20)` et observez.  
2. **Comparer avec un modèle plus simple :** testez une `LogisticRegression` sur le même dataset.  
3. **Visualiser les erreurs :** affichez quelques chiffres mal classés.

```python
import numpy as np
misclassified = np.where(... != ...)[0][:4]
for idx in misclassified:
    plt.imshow(...[idx].reshape(8,8), cmap='gray')
    plt.title(f"Vrai: {...[idx]} / Prédit: {...[idx]}")
    plt.show()
```
