# TP4 – Découverte des réseaux de neurones convolutifs (CNN)

## Objectifs
- Comprendre la structure d’un réseau convolutionnel (CNN)
- Comparer CNN et MLP sur des données d’images (MNIST)
- Visualiser les filtres et cartes de caractéristiques (feature maps)
- Identifier les hyperparamètres essentiels du CNN

Durée : 2h–2h30  
Environnement : **Google Colab** (aucune installation nécessaire)

---

## Étape 0 – Introduction : Keras vs PyTorch

### 0.1 Pourquoi ce choix ?

Il existe plusieurs bibliothèques pour créer des réseaux de neurones :

| Framework | Niveau d’abstraction | Particularité |
|------------|---------------------|----------------|
| **Keras / TensorFlow** | Haut niveau | Simple, rapide à utiliser, idéal pour débuter |
| **PyTorch** | Plus bas niveau | Plus flexible, plus proche du fonctionnement interne |
| **Scikit-learn** | Très haut niveau | Pour modèles simples (KNN, MLP, etc.) |

Dans ce TP, nous utilisons **Keras (avec TensorFlow en arrière-plan)** :  

**Q0.1.** Pourquoi Keras est-il plus adapté à un premier TP sur les CNN ?  

**Q0.2.** Que changerait l’utilisation de PyTorch ?  

---

## Étape 1 – Préparation et exploration du dataset

### Vérifions d’abord notre environnement Colab :

```python
import tensorflow as tf
print("Version TensorFlow :", tf.__version__)
print("GPU disponible :", tf.config.list_physical_devices('GPU'))
```

> Si "GPU disponible : []", allez dans **Exécution → Modifier le type d’exécution → Accélérateur matériel : GPU**.

**Q0.3.** Que change concrètement l’utilisation d’un GPU par rapport au CPU lors de l’entraînement ?  
📚 *Aide : [Keras – GPU guide](https://www.tensorflow.org/guide/gpu)*  

---

### Chargement du dataset MNIST

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Taille du dataset :", X_train.shape, X_test.shape)

# TODO : normaliser les pixels entre 0 et 1
# X_train = ...
# X_test = ...

# TODO : afficher 4 exemples du jeu d'entraînement avec plt.imshow()
```
📚 *Aide : [Keras MNIST Dataset](https://keras.io/api/datasets/mnist/)*  

**Q1.** Quelle est la taille d’une image d’entrée ?  
**Q2.** Pourquoi normaliser les pixels entre 0 et 1 ?  
**Q3.** Quelle est la dimension de sortie du modèle avant d’ajouter le canal (28x28 -> ?)**  
**Q4.** Pourquoi doit-on “reshaper” les données avant de les donner à un CNN ?  

---

### Mise en forme pour le CNN

```python
# TODO : reshape les images pour ajouter le canal (28, 28, 1)
# TODO : encoder les labels (one-hot encoding)

# Exemple attendu :
# X_train = X_train.reshape(-1, 28, 28, 1)
# y_train_cat = to_categorical(y_train, 10)
```
📚 *Aide : [to_categorical – Keras utils](https://keras.io/api/utils/python_utils/#to_categorical-function)*  

---

## Étape 2 – Modèle MLP de référence

On commence par un MLP simple pour avoir un point de comparaison.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

mlp = Sequential([
    Flatten(input_shape=(28,28,1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO : entraîner le modèle sur 5 epochs
# history_mlp = mlp.fit(...)
```
📚 *Aide : [Sequential Model – Keras](https://keras.io/api/models/sequential/)*  
📚 *Aide : [Model.fit() – Keras](https://keras.io/api/models/model_training_apis/#fit-method)*  

**Q5.** Quelle précision obtenez-vous sur le train et le test ?  
**Q6.** Pourquoi le MLP possède-t-il beaucoup de paramètres ?  
**Q7.** Quelles sont les limites du MLP pour des images ?  

---

## Étape 3 – Construction d’un petit CNN

Créons un CNN simple à 2 couches convolutionnelles.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D

cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO : entraîner le CNN sur 5 epochs, batch_size=128
# history_cnn = cnn.fit(...)
```
📚 *Aide : [Conv2D – Keras](https://keras.io/api/layers/convolution_layers/convolution2d/)*  
📚 *Aide : [MaxPooling2D – Keras](https://keras.io/api/layers/pooling_layers/max_pooling2d/)*  
📚 *Aide : [Flatten – Keras](https://keras.io/api/layers/reshaping_layers/flatten/)*  

**Q8.** Combien de couches convolutives et de pooling contient ce réseau ?  
**Q9.** Pourquoi a-t-on besoin d’une couche `Flatten()` avant les couches denses ?  
**Q10.** Quelle différence vois-tu entre ce modèle et le MLP précédent ?  
**Q11.** Compare la précision obtenue avec le CNN et le MLP.  

---

## Étape 4 – Visualisation des performances

### Tracer les courbes d’apprentissage

```python
import matplotlib.pyplot as plt

# TODO : tracer la précision du train et du test pour le CNN
plt.plot(..., label='train')
plt.plot(..., label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Évolution de la précision du CNN')
plt.legend()
plt.show()
```
📚 *Aide : [Model.fit() history object – Keras](https://keras.io/api/models/model_training_apis/#fit-method)*  

**Q12.** Comment évolue la précision sur le train et le test ?  
**Q13.** Vois-tu un signe d’overfitting ? (indices : écart entre train et test)  
**Q14.** Quelles solutions pourrais-tu proposer pour le limiter ? (Dropout, Data Augmentation, etc.)  

---

## Étape 5 – Exploration visuelle : filtres et feature maps

### Afficher les filtres appris dans la première couche
```python
weights, biases = cnn.layers[0].get_weights()
print("Forme des filtres :", weights.shape)
```
📚 *Aide : [get_weights() – Keras](https://keras.io/api/models/model_training_apis/#get_weights-method)*  

### Visualiser les cartes de caractéristiques (feature maps)
```python
from tensorflow.keras import Model

# TODO : choisir une image test
sample = X_test[0].reshape(1,28,28,1)

# TODO : créer un modèle qui sort la première couche convolutionnelle
model_layer1 = Model(inputs=cnn.inputs, outputs=cnn.layers[0].output)

feature_maps = model_layer1.predict(sample)

# TODO : afficher les 6 premières cartes
for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(feature_maps[0,:,:,i], cmap='gray')
    plt.axis('off')
plt.show()
```
📚 *Aide : [Functional API – Model() Keras](https://keras.io/guides/functional_api/)*  

**Q15.** Que représentent les “poids” de la première couche ?  
**Q16.** Que montrent les “feature maps” ?  
**Q17.** Les filtres apprennent-ils tous la même chose ? Pourquoi ?  

---

## Étape 6 – Comparaison CNN vs MLP

| Aspect | MLP | CNN |
|---------|-----|-----|
| Entrée | ... | ... |
| Poids | ... | ... |
| Structure | ... | ... |
| Précision | ... | ... |
| Robustesse | ... | ... |

**Q18.** Pourquoi le CNN généralise-t-il mieux que le MLP ?  
**Q19.** Que se passerait-il si on ajoutait plus de couches convolutionnelles ?  
**Q20.** Dans quels cas un MLP resterait-il plus adapté qu’un CNN ?  

---

## Bonus : aller plus loin

1. Ajoute une couche **Dropout(0.25)** après le Flatten.  
2. Relance l’entraînement et observe l’effet sur la précision.  
3. Essaie un **filtre 5×5** au lieu de 3×3.  
4. Compare le nombre de paramètres entre MLP et CNN :  
   ```python
   cnn.summary()
   mlp.summary()
   ```
📚 *Aide : [Dropout – Keras layer](https://keras.io/api/layers/regularization_layers/dropout/)*  

**Q21.** Que remarques-tu sur la complexité des modèles ?  
**Q22.** Quel est le lien entre nombre de paramètres et risque d’overfitting ?  

---

## Synthèse

Ce TP vous a permis de :
- Construire un **CNN** avec Keras.
- Comprendre la différence entre **MLP** et **CNN**.
- Visualiser ce que le réseau apprend réellement (feature maps).
- Relier les **hyperparamètres** à la performance et à la généralisation.

**Message clé :**
> Un CNN n’a pas besoin qu’on lui dise “ce qu’est un bord” :  
> il l’apprend automatiquement à travers ses filtres.
