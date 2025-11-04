# TP5 – Régularisation et généralisation dans les CNN (CIFAR‑10)

## Objectifs
- Appliquer un CNN à un nouveau jeu de données plus réaliste : CIFAR‑10.
- Comprendre et comparer plusieurs méthodes de régularisation : Dropout, L2, BatchNorm, Data Augmentation, EarlyStopping.
- Observer leurs effets sur les performances et les courbes d’apprentissage.

Environnement : Google Colab (TensorFlow / Keras) 

**Remarque : Sans GPU activité, les temps seront très longs !**

---

## Étape 0 – Chargement et exploration du dataset

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# Chargement
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalisation
X_train = X_train.astype("float32")/255.0
X_test  = X_test.astype("float32")/255.0

# One-hot
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)
```

**Q0.1.** Quelle est la taille et la structure des images ?  
**Q0.2.** Combien de classes contient CIFAR‑10 ?  

📚 *Aide : [Keras CIFAR‑10 Dataset](https://keras.io/api/datasets/cifar10/)*  

### Visualisation d’exemples
```python
labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(X_train[i])
    plt.title(labels[int(y_train[i])])
    plt.axis("off")
plt.show()
```
**Q0.3.** Quelle différence majeure avec MNIST remarques‑tu (taille, couleur, complexité) ?  

---

## Étape 1 – CNN de base (sans régularisation)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

base = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

base.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_base = base.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20, batch_size=128, verbose=1
)
```

**Q1.** Quelle précision obtiens‑tu sur le train et sur le test ?  
**Q2.** Observe les courbes d’accuracy : y a‑t‑il un overfitting ? Si oui, à quel moment se produit-il ?  

```python
plt.plot(history_base.history['accuracy'],label='train')
plt.plot(history_base.history['val_accuracy'],label='val')
plt.legend(); plt.title("CNN sans régularisation")
plt.show()
```

---

## Étape 2 – Dropout

```python
from tensorflow.keras.layers import Dropout

drop = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

drop.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_drop = drop.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20, batch_size=128, verbose=1
)
```

**Q3.** Qe signiie le Dropout ? Compare les courbes train/val : l’écart diminue‑t‑il ?  
**Q4.** Que se passe‑t‑il si le taux de Dropout est trop élevé ?  

📚 *Aide : [Dropout – Keras](https://keras.io/api/layers/regularization_layers/dropout/)*

---

## Étape 3 – Régularisation L2

```python
from tensorflow.keras.regularizers import l2

l2_model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu',kernel_regularizer=l2(0.001)),
    Dense(10,activation='softmax')
])

l2_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_l2 = l2_model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20, batch_size=128, verbose=1
)
```

**Q5.** Qu'est ce que la régularisation L2 ? Quel impact : L’écart train/test diminue‑t‑il ?  
**Q6.** Quel effet aurait une régularisation trop forte ?  

📚 *Aide : [Regularizers – Keras](https://keras.io/api/layers/regularizers/)*

---

## Étape 4 – Batch Normalization

```python
from tensorflow.keras.layers import BatchNormalization

bn = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

bn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_bn = bn.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20, batch_size=128, verbose=1
)
```

**Q7.** Qu'est ce que la Batch Normalization ? La convergence est‑elle plus rapide ou plus stable ?  
**Q8.** Pourquoi cette normalisation aide‑t‑elle ?  

📚 *Aide : [BatchNormalization – Keras](https://keras.io/api/layers/normalization_layers/batch_normalization/)*

---

## Étape 5 – Data Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow(X_train, y_train_cat, batch_size=128)

aug = tf.keras.models.clone_model(base)
aug.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_aug = aug.fit(
    train_gen,
    validation_data=(X_test, y_test_cat),
    epochs=20, verbose=1
)
```

**Q9.** Qu'est ce que la Data Augmentation ? Pourquoi cette méthode réduit‑elle l’overfitting ?  
**Q10.** Quelles transformations semblent les plus efficaces ?  **Si c'est trop long, réduisez éventuellement le nombre d'Epochs**

📚 *Aide : [ImageDataGenerator – Keras](https://keras.io/api/preprocessing/image/)*

---

## Étape 6 – Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

es = tf.keras.models.clone_model(base)
es.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_es = es.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50, batch_size=128, callbacks=[early], verbose=1
)
```

**Q11.** Combien d’époques sont réellement effectuées ?  
**Q12.** Qu'est ce que l'Early Stopping ? Quel intérêt par rapport à un nombre fixe d’époques ?  

📚 *Aide : [EarlyStopping – Keras](https://keras.io/api/callbacks/early_stopping/)*

---

## Étape 7 – Comparaison et analyse

| Méthode | Accuracy (train) | Accuracy (test) | Écart réduit ? | Commentaire |
|----------|------------------|------------------|----------------|--------------|
| Aucune | … | … | ☐ Oui ☐ Non |  |
| Dropout | … | … | ☐ Oui ☐ Non |  |
| L2 | … | … | ☐ Oui ☐ Non |  |
| BatchNorm | … | … | ☐ Oui ☐ Non |  |
| Data Aug | … | … | ☐ Oui ☐ Non |  |
| Early Stop | … | … | ☐ Oui ☐ Non |  |

**Q13.** Quelle méthode te semble la plus efficace ?  
**Q14.** Peut‑on les combiner ? Si oui, comment ?  

---

## Synthèse

Ce TP t’a permis de :
- Expérimenter les principales techniques de régularisation sur un dataset réaliste (CIFAR‑10).  
- Observer leurs effets concrets sur les performances et la stabilité.  
- Comprendre que généraliser, c’est trouver l’équilibre entre capacité et contrôle.
