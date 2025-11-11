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
- Q1. Quelles différences fondamentales distinguent cet apprentissage par essai-erreur d’un apprentissage supervisé classique ?  
- Q2. Pourquoi la notion de retour d’expérience (reward/penalty) est-elle essentielle dans ce type d’apprentissage ?  
- Q3. Citez deux exemples réels où un agent, humain ou machine, apprend de manière similaire à ce principe.  
- Q4. Quels sont les avantages et les limites de ce mode d’apprentissage ?  

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

