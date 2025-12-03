# TweetClassifier — Multimodal Twitter Classifier

Projet: Classificateur multimodal (texte + métadonnées) pour tweets basé sur PyTorch et HuggingFace.

## 🌟 Résumé
Ce dépôt contient une implémentation d'un classificateur multimodal pour des tweets qui combine:
- Un encodeur Transformer (XLM-RoBERTa) sur le texte
- Un MLP léger sur 8 features numériques (métadonnées d'utilisateur)
- Une couche de fusion qui concatène l'embedding CLS du transformer et la sortie du MLP

Le but est de produire une prédiction binaire (0/1).

## 📁 Structure du projet
- `dataset.py` : Dataset PyTorch (tokenization + conversion metadata -> tensors).
- `model.py` : `MultimodalTweetClassifier` (Transformer + metadata MLP + classifier)
- `train_multimodal.py` : Script d'entraînement avec validation et sauvegarde du meilleur modèle
- `predict_kaggle.py` : Script de prédiction pour `kaggle_test.jsonl` (sauvegarde `multimodal_transformer.csv`)
- `config.py` : Dataclass `TrainingConfig` pour centraliser les hyperparamètres
- `baseline.ipynb` : Notebook d'exploration et prétraitement

> Les fichiers de données (`train.jsonl`, `kaggle_test.jsonl`) et les modèles entraînés (`best_multimodal_model.pt`, `scaler.pkl`) **ne sont pas** inclus dans le dépôt (voir `.gitignore`).

## ⚙️ Prérequis
- Python 3.8+ (ou 3.10/3.11 selon votre environnement)
- Recommandé: GPU CUDA ou MPS (Mac)

Installer les dépendances :

```bash
# Créez un environnement virtuel (optionnel mais recommandé)
python3 -m venv venv
source venv/bin/activate
# Installer les dépendances (voir requirements.txt)
pip install -r requirements.txt
```

## 🔧 Installation (rapide)
```bash
# Si vous n'avez pas encore cloné le repo
git clone https://github.com/ComeRochas/TweetClassifier.git
cd TweetClassifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Entraînement
Le script `train_multimodal.py` inclut la tokenization, la normalisation des métadonnées, la séparation train/val, et le training.

Exemple d'entraînement (rapide):

```bash
python3 train_multimodal.py --num_epochs 1 --batch_size 8
```

Options utiles (CLI)
- `--batch_size` : taille de batch (par défaut 16)
- `--num_epochs` : nombre d'époques (par défaut 4)
- `--lr_transformer` : LR pour transformer (par défaut 2e-5)
- `--lr_head` : LR pour meta head et classifier (par défaut 1e-3)
- `--max_length` : longueur max des tokens (par défaut 160)
- `--freeze_transformer` : gèle le transformer (ne l'entraîne pas)

Exemple avec freeze:

```bash
python3 train_multimodal.py --freeze_transformer --num_epochs 3
```

> Astuce : commencez par geler le transformer (`--freeze_transformer`) pour valider la boucle d'entraînement rapidement, puis débloquez-le pour fine-tuning si besoin.

## 🧪 Prédiction / Soumission Kaggle
Une fois le modèle entraîné (`best_multimodal_model.pt`), le script `predict_kaggle.py`:

- Charge le scaler (scaler.pkl)
- Tokenize + scale les features
- Prépare un fichier `multimodal_transformer.csv` avec deux colonnes : `ID`, `Prediction`

Exemple:

```bash
python3 predict_kaggle.py
```

## 📌 Notes sur les hyperparamètres
- Le fichier `config.py` contient `TrainingConfig` qui regroupe les hyperparamètres par défaut.
- On différencie les LR des paramètres du transformer (petit) et du head (plus haut).
- Par défaut `max_length` est 160 — les stats sur vos tweets montrent un médian ≈ 55, mean ≈ 58, 99e percentile ≈ 127, ainsi 160 est conservateur (évite la troncature pour la plupart).

## 🗂️ Données
Le format attendu par les scripts :
- `train.jsonl` : JSON Lines, chaque ligne contient un tweet, avec des champs user.* et éventuellement `label`: 0 / 1
- `kaggle_test.jsonl` : similaire mais sans `label`; contient `challenge_id` pour la soumission

> Les étapes d'extraction / scaling sont déjà implémentées dans `train_multimodal.py` et `predict_kaggle.py`.

## 🔁 Reproductibilité
- Les seeds sont fixées via `cfg.seed` depuis `config.py`

## 🛠️ Développement & contributions
- Ajoutez des issues / PR si vous souhaitez améliorer les datasets, features, ou le modèle (ex: combiner un finetuning progressif, scheduler, etc.)

## 📜 Licence
- Ajoutez la licence si vous voulez rendre le projet public; actuellement pas de licence définie.

---

Si vous voulez, je peux :
- Ajouter un `README` en anglais en plus
- Créer un `requirements.txt` (je l'ai déjà ajouté) et l'installer automatiquement
- Ajouter une action GitHub CI pour tests/format
- Ajouter un `Makefile` ou un wrapper `run.sh`

N'hésitez pas à me dire ce que vous préférez !
