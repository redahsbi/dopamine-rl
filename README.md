# Dopaminergic Reinforcement Learning

En découvrant les bases du reinforcement learning, je suis tombé sur quelque chose qui m'a vraiment surpris : l'algorithme TD-learning calcule à chaque pas une quantité appelée **Reward Prediction Error (RPE)**, et cette même quantité a été mesurée expérimentalement dans le cerveau de singes par Wolfram Schultz en 1997. Les neurones dopaminergiques ne répondent pas à la récompense elle-même, mais à l'**erreur de prédiction** de cette récompense.

J'ai voulu vérifier ça par moi-même en codant un agent simple et en observant ce signal au fil de l'entraînement.

---

## L'idée

Le RPE (noté δ) se calcule ainsi :

```
δ = r + γ · V(s') − V(s)
```

- **δ > 0** → l'agent reçoit mieux que prévu → burst de dopamine
- **δ < 0** → l'agent reçoit moins que prévu → chute de dopamine
- **δ ≈ 0** → exactement ce qui était prévu → rien ne se passe

C'est exactement la règle de mise à jour du Q-Learning :

```
Q(s, a) += α · δ
```

Le signal qui fait apprendre l'agent et le signal dopaminergique biologique sont donc la même chose. C'est ce que ce projet essaie de montrer visuellement.

---

## L'environnement

J'ai créé une grille 5×5 inspirée des paradigmes de conditionnement classique utilisés en neurosciences :

```
S  .  .  .  .
.  .  .  .  .
.  .  .  .  .
.  .  .  .  .
X  .  .  C  G
```

| Case | Rôle | Récompense |
|------|------|-----------|
| `S` | Départ | — |
| `G` | But — Stimulus Inconditionnel (US) | +10 |
| `C` | Zone CS — Stimulus Conditionnel | +0.5 |
| `X` | Piège | −5 |
| `.` | Case vide | −0.1 (coût de déplacement) |

La distinction CS/US est volontaire : elle permet d'observer le **transfert temporel** du signal dopaminergique, le résultat principal du projet.

Un épisode se termine quand l'agent atteint G, ou après 100 pas maximum.

---

## Architecture du code

```
dopamine-rl/
├── src/
│   ├── agent.py         # L'agent Q-Learning avec calcul explicite du RPE
│   ├── environment.py   # La grille Pavlovienne
│   ├── train.py         # Boucle d'entraînement sur 500 épisodes
│   └── visualize.py     # Génération des graphiques
├── results/
│   ├── agent.json            # Q-table finale sauvegardée
│   ├── results.json          # Logs complets de l'entraînement
│   ├── learning_curve.png
│   ├── rpe_dynamics.png
│   ├── temporal_transfer.png
│   ├── value_maps.png
│   └── summary_dashboard.png
├── requirements.txt
└── README.md
```

Le cœur du projet est dans `agent.py`. La méthode `compute_rpe()` calcule δ à chaque pas, et `update()` l'utilise pour mettre à jour la Q-table tout en enregistrant l'historique du signal.

```python
def compute_rpe(self, state, action, reward, next_state, done):
    v_current = self.Q[state, action]
    v_next = 0.0 if done else np.max(self.Q[next_state])
    rpe = reward + self.gamma * v_next - v_current
    return rpe

def update(self, state, action, reward, next_state, done):
    rpe = self.compute_rpe(state, action, reward, next_state, done)
    self.Q[state, action] += self.alpha * rpe
    self.rpe_history.append(rpe)
    return rpe
```

---

## Installation

```bash
git clone https://github.com/redahsbi/dopamine-rl.git
cd dopamine-rl
pip install -r requirements.txt
```

**Lancer l'entraînement :**
```bash
cd src
python3 train.py
```

**Générer les graphiques :**
```bash
python3 visualize.py
```

---

## Résultats

### Graphique 1 — Courbe d'apprentissage

![Learning Curve](results/learning_curve.png)

La récompense moyenne par épisode sur 500 épisodes. Au début l'agent erre au hasard et tombe souvent dans le piège (récompense ≈ 2). Vers l'épisode 150-200 la courbe décolle — la Q-table a accumulé assez d'informations pour guider l'agent efficacement. Elle se stabilise autour de **9.79**, ce qui correspond au chemin optimal de 8 pas (10 − 8×0.1 = 9.2, légèrement dépassé car l'agent trouve parfois des raccourcis).

---

### Graphique 2 — Le signal RPE au fil de l'entraînement

![RPE Dynamics](results/rpe_dynamics.png)

C'est le graphique le plus intéressant biologiquement. Il montre δ moyen par épisode (rouge = δ positif / burst, bleu = δ négatif / dip) et la décroissance de ε en dessous.

On observe trois phases :

- **Épisodes 1–50** : la Q-table est vide, l'agent n'attend rien → peu de surprise → δ faible. Pas parce qu'il sait, mais parce qu'il n'a pas encore d'attentes.
- **Épisodes 50–200** : les attentes commencent à se former → l'écart entre ce qui est prévu et ce qui arrive devient grand → pic de δ. C'est la phase d'apprentissage actif.
- **Épisodes 200–500** : les prédictions deviennent précises → δ → 0. L'apprentissage converge.

Le panneau du bas montre ε qui diminue de 1.0 à ~0.08 : l'agent passe progressivement de l'exploration aléatoire à l'exploitation de ce qu'il a appris.

---

### Graphique 3 — Transfert temporel du signal dopaminergique

![Temporal Transfer](results/temporal_transfer.png)

C'est le résultat principal du projet, et celui qui m'a le plus surpris.

On trace deux signaux séparément :
- **Orange** : δ moyen quand l'agent atteint G (le but, stimulus inconditionnel)
- **Rouge pointillé** : δ moyen quand l'agent passe par la zone C (stimulus conditionnel)

Au début, seul G surprend l'agent (courbe orange haute). La zone C ne l'intéresse pas.

Après ~150 épisodes, l'agent a appris que **C précède toujours G**. Arriver en C devient alors la vraie nouvelle information — G n'est plus une surprise car il était déjà prédit depuis C. Le signal δ **remonte dans le temps** vers le prédicteur le plus précoce.

C'est exactement ce que Schultz a observé chez le singe : au début les neurones dopaminergiques s'activent quand le jus arrive. Après conditionnement, ils s'activent quand la lumière qui précède le jus s'allume — et plus du tout quand le jus arrive.

---

### Graphique 4 — Évolution de la fonction de valeur V(s)

![Value Maps](results/value_maps.png)

10 instantanés de la fonction de valeur V(s) = maxₐ Q(s,a) au fil de l'entraînement, affichés comme des cartes de chaleur sur la grille. Les couleurs chaudes indiquent une valeur élevée (l'agent espère une grande récompense future depuis cette case), les couleurs froides une valeur basse ou négative.

Au début tout est uniforme. La connaissance se propage **à rebours depuis G** : les cases voisines de G apprennent en premier, puis leurs voisines, etc. À l'épisode 500 la grille est entièrement éclairée et l'agent peut suivre le gradient de valeur depuis n'importe quelle position pour rejoindre G en 8 pas.

Le piège X (coin bas-gauche) reste systématiquement froid — l'agent a appris à l'éviter.

---

### Dashboard récapitulatif

![Dashboard](results/summary_dashboard.png)

Vue d'ensemble en une seule image : courbe d'apprentissage, dynamique du RPE, transfert CS→US, et trois instantanés de la value map (début, épisode ~55, fin).

---

## Références

- Schultz, W., Dayan, P., & Montague, P.R. (1997). *A neural substrate of prediction and reward.* Science, 275(5306), 1593–1599.
- Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## Auteur

**Reda Hasbi** — 1ère année, ENSEIRB-MATMECA Informatique  
Oracle Cloud Infrastructure 2025 AI Foundations Associate

*Projet réalisé dans le cadre d'une candidature à un stage à l'équipe MNEMOSYNE (Inria Bordeaux), sur la modélisation cérébrale et les systèmes cognitifs intégratifs.*