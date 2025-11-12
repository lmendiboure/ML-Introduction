# Vers une mise en oeuvre de l'apprentissage par renforcement (RL)

## Partie 1 – Comprendre le Reinforcement Learning

Cette première partie vise à introduire le concept fondamental du Reinforcement Learning (RL).  
L’objectif est que vous puissiez comprendre le fonctionnement général d’un agent apprenant avant de le programmer.

---

## 1. Comprendre l’apprentissage par essai-erreur

Un agent de Reinforcement Learning apprend par interaction avec son environnement : il effectue une action, observe le résultat, et ajuste sa stratégie en conséquence.

Prenons un exemple concret : un robot mobile doit apprendre à atteindre une cible sans heurter d’obstacles.  
Au départ, il agit aléatoirement. À chaque collision, il reçoit une pénalité ; à chaque réussite, une récompense.  
Au fil du temps, il apprend quelles séquences d’actions mènent à la réussite.

Ce principe s’applique aussi à des contextes humains : un individu ajuste son comportement selon les conséquences de ses décisions.

### Questions de réflexion
- **Q1.** Quelles différences fondamentales distinguent cet apprentissage par essai-erreur d’un apprentissage supervisé classique ?  
- **Q2.** Pourquoi la notion de retour d’expérience (reward/penalty) est-elle essentielle dans ce type d’apprentissage ?  
- **Q3.** Citez deux exemples réels où un agent, humain ou machine, apprend de manière similaire à ce principe.  
- **Q4.** Quels sont les avantages et les limites de ce mode d’apprentissage ?  

---

## 2. Le schéma général du RL

```text
        ┌────────────────────┐
        │   Environnement    │
        └────────┬───────────┘
                 │
            observe état s
                 │
                 ▼
          ┌────────────┐
          │   Agent    │
          └────┬───────┘
               │
     choisit action a
               │
               ▼
        reçoit récompense r
        + nouvel état s'
```

Chaque interaction suit le même cycle :

1. L’agent observe un état de l’environnement.  
2. Il choisit une action selon sa politique actuelle.  
3. L’environnement renvoie une récompense et un nouvel état.  
4. L’agent met à jour sa politique à partir de cette expérience.

Le but du RL est d’apprendre une politique optimale π*(s) = a*, maximisant la somme des récompenses cumulées à long terme.

---

### Questions de réflexion

- **Q5.** Identifiez les trois informations principales échangées entre l’agent et l’environnement.  
- **Q6.** En quoi le rôle de la récompense est-il différent d’une simple étiquette (*label*) dans l’apprentissage supervisé ?  
- **Q7.** Quelles difficultés peut rencontrer un agent lorsqu’il doit apprendre sans récompense immédiate ?  

---

## 3. Comparaison entre apprentissage supervisé et apprentissage par renforcement

| **Caractéristique** | **Apprentissage supervisé** | **Apprentissage par renforcement** |
|----------------------|-----------------------------|------------------------------------|
| **Données** | Exemples annotés (x, y) | Expériences issues d’interactions |
| **Objectif** | Prédire la sortie correcte | Trouver une politique optimale |
| **Retour d’information** | Immédiat et explicite | Délayé et parfois incertain |
| **Exemple typique** | Reconnaissance d’image | Jeu, robotique, contrôle |

---

### Questions de réflexion

- **Q8.** Pourquoi le RL est-il souvent plus difficile à entraîner que l’apprentissage supervisé ?  
- **Q9.** Expliquez en quoi la notion d’exploration est centrale dans le RL.  
- **Q10.** Donnez un exemple de situation où le RL serait plus adapté que le supervisé.  

---

## 4. Expérimentation simple : un mini-monde

Créons un petit environnement à 3 états et 2 actions possibles.  
Cet exemple illustre comment un agent peut évaluer les conséquences de ses choix.

```python
import numpy as np

n_states, n_actions = 3, 2
R = np.array([[-1, 0],
              [10, -10],
              [0, 0]])
Q = np.zeros((n_states, n_actions))

alpha, gamma = 0.5, 0.9

for episode in range(20):
    s = 0
    done = False
    while not done:
        a = np.random.choice([0,1])
        reward = R[s, a]
        s2 = min(s + 1, 2) if a == 0 else max(s - 1, 0)
        Q[s,a] = Q[s,a] + alpha * (reward + gamma * np.max(Q[s2]) - Q[s,a])
        if s2 == 2:
            done = True
        s = s2

print(Q)
```

### Questions d’analyse

- **Q11.** Que représente chaque cellule du tableau `Q` ?
- **Q12.** Pourquoi certaines valeurs deviennent-elles positives et d’autres négatives ?
- **Q13.** Quelle action semble la plus intéressante dans l’état 0 ?
- **Q14.** Que se passe-t-il si l’on modifie la récompense finale (par ex. `+10 → +1`) ?
- **Q15.** Pourquoi le paramètre γ (*gamma*) est-il important pour valoriser les récompenses futures ?

---

## 5. Exemples d’applications concrètes du Reinforcement Learning

| **Domaine**   | **Exemple concret**            | **Objectif de l’agent**                          |
|---------------|--------------------------------|--------------------------------------------------|
| Jeux vidéo    | AlphaGo, Atari, Dota 2         | Maximiser la probabilité de victoire             |
| Robotique     | Bras robotisé, drones          | Réaliser une tâche motrice précise               |
| Transport     | Feux de circulation intelligents| Fluidifier le trafic et réduire l’attente        |
| Énergie       | Régulation chauffage / réseau  | Optimiser la consommation énergétique            |
| Santé         | Traitement personnalisé        | Adapter la thérapie au profil du patient         |

### Questions de réflexion

- **Q16.** Quels points communs retrouve-t-on entre ces différentes applications ?
- **Q17.** Dans quels domaines le RL pourrait-il encore être sous-exploité ?
- **Q18.** Donnez un exemple où le RL ne serait pas adapté, et justifiez votre réponse.
- **Q19.** Quelles sont, selon vous, les principales limites techniques (voire éthiques si vous avez des idées) du RL ? Quelles difficultés dans sa mise en oeuvre ?

---

## Partie 2 - Application du Reinforcement Learning à un environnement discret - Environnement Taxi-v3  

Cette seconde partie a pour objectif de vous faire implémenter et comprendre l’algorithme de Q-learning tabulaire à travers l’environnement `Taxi-v3` de la bibliothèque `gymnasium`.  
Vous observerez comment un agent apprend à interagir efficacement avec un environnement déterministe et discret.

## 1. Découverte et configuration de l’environnement


```python
# Install dependencies (Colab)
!pip -q install gymnasium matplotlib numpy
import gymnasium as gym
env = gym.make("Taxi-v3")
obs, info = env.reset()
env.render()
print("Number of states:", env.observation_space.n)
print("Number of actions:", env.action_space.n)
```

L’environnement représente une grille 5x5 :
- 4 emplacements fixes (R, G, B, Y)
- un taxi
- un passager
- un objectif

L’agent reçoit :
- +20 lorsqu’il dépose correctement le passager,
- -1 par action pour encourager les stratégies efficaces,
- -10 en cas de dépôt incorrect.

### Questions 

- **Q1.** Décrivez la structure générale de cet environnement.

- **Q2.** Combien d’états distincts peuvent exister ? Pourquoi ce nombre est-il élevé ?

- **Q3.** Quelles sont les actions possibles à chaque étape ?

- **Q4.** Que se passe-t-il lorsqu’une action est illégale (ex : déposer sans passager) ?

## 2. Implémentation du Q-learning

L’agent construit progressivement une table de valeurs Q permettant d’estimer la valeur de chaque action dans chaque état.
La mise à jour suit la formule : **Q(s,a)←Q(s,a)+α[r+γa′max​Q(s′,a′)−Q(s,a)]**

```python
import numpy as np

n_states  = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

alpha = 0.1      # learning rate
gamma = 0.99     # discount factor
epsilon = 1.0    # exploration rate
eps_min, eps_decay = 0.05, 0.995

rewards = []

for episode in range(1000):
    s, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
        s2, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        # Q-learning update
        Q[s,a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s,a])
        s = s2
        total_reward += r

    epsilon = max(eps_min, epsilon * eps_decay)
    rewards.append(total_reward)
```
### Questions 

- **Q5.** Que représente la Q-table dans cet algorithme ?

- **Q6.** Quel rôle jouent les paramètres alpha, gamma et epsilon ?

- **Q7.** Pourquoi diminue-t-on progressivement la valeur de epsilon ?

- **Q8.** Que signifie une grande valeur de Q[s,a] ?

## 3. Visualisation et évaluation

```python
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Taxi-v3 Learning Curve")
plt.grid(True)
plt.show()
```
```python
# Evaluate the learned policy
obs, info = env.reset()
done = False
steps = 0
while not done and steps < 50:
    a = np.argmax(Q[obs])
    obs, r, terminated, truncated, info = env.step(a)
    env.render()
    done = terminated or truncated
    steps += 1
```
### Questions 

- **Q9.** Comment évoluent les récompenses moyennes au fil du temps ?

- **Q10.** Le comportement final du taxi est-il optimal ? Justifiez.

- **Q11.** Pourquoi l’apprentissage reste-t-il imparfait même après 1000 épisodes ?

## 4. Expérimentation : impact des hyperparamètres

```python
configs = [(0.1, 0.99, 1.0), (0.3, 0.9, 0.5), (0.5, 0.95, 0.1)]

for (alpha, gamma, epsilon) in configs:
    Q = np.zeros((n_states, n_actions))
    for ep in range(500):
        s, info = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[s])
            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            Q[s,a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s,a])
            s = s2
```
### Questions 

- **Q12.** Quels paramètres accélèrent la convergence ?

- **Q13.** Que se passe-t-il si gamma est trop faible ?

- **Q14.** Pourquoi un epsilon trop petit peut-il ralentir l’apprentissage ?

## 5. Expérimentation avancée : modification des récompenses

Essayez maintenant d’ajuster la récompense des actions inutiles (*-1* → *-3*) et observez l’effet.

```python
Q = np.zeros((n_states, n_actions))
alpha, gamma, epsilon = 0.1, 0.99, 0.3

for ep in range(300):
    s, info = env.reset()
    done = False
    while not done:
        a = np.argmax(Q[s]) if np.random.rand() > epsilon else env.action_space.sample()
        s2, r, terminated, truncated, info = env.step(a)
        if r == -1:
            r = -3  # penalize inefficient moves
        Q[s,a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s,a])
        s = s2
        done = terminated or truncated
```

- **Q15.** Que se passe-t-il lorsque les actions inutiles sont plus fortement pénalisées ?

- **Q16.** L’apprentissage devient-il plus rapide ou plus risqué ?

- **Q17.** Qu’est-ce que le reward shaping et quels en sont les risques ?

---

## Partie 3 - Sujet bonus – FrozenLake-v1 et CartPole-v1

Explorer un environnement différent pour observer les limites du Q-learning :
- FrozenLake-v1 (stochastique) → gestion de l’incertitude, exploration.
- CartPole-v1 (continu) → besoin d’un réseau neuronal (DQN).

## 1. Configuration FrozenLake 
```python
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=True)
obs, info = env.reset()
env.render()
```
### Questions 

- **Q1.** Quelle différence majeure observe-t-on avec Taxi-v3 ?

- **Q2.** Pourquoi l’environnement est-il plus difficile à maîtriser ?

- **Q3.** Le Q-learning converge-t-il de manière stable ?

## 2.Configuration CartPole 
```python
!pip install tensorflow keras gymnasium matplotlib
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
```

**Implémentation d’un DQN minimal**

L’objectif est de stabiliser le pôle en apprenant une politique à partir de l’observation continue.

```python
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(n_actions, activation='linear')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse')
```
### Questions 

- **Q4.** Pourquoi un réseau de neurones est-il nécessaire ici ?

- **Q5.** Quelle différence fondamentale avec le Q-learning tabulaire ?

- **Q6.** Comment juger de la stabilité de la politique apprise ?

- **Q7.** En quoi la complexité de l’environnement conditionne-t-elle le choix de l’algorithme de Reinforcement Learning ?
