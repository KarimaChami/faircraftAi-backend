# 🌟 FairCraft AI

**FairCraft AI** est une plateforme innovante conçue pour aider les artisans à estimer de manière juste et compétitive le prix de leurs créations faites à la main. En s'appuyant sur des données réelles du marché (ex: Etsy) et des modèles de Machine Learning performants (RandomForest, XGBoost), l'application fournit des recommandations de prix intelligentes et explicables.

---

## 🚀 Fonctionnalités Principales

- 📊 **Scraping et Analyse du Marché :** Collecte automatisée de produits artisanaux similaires sur des plateformes comme Etsy pour alimenter la base de données de référence.
- 🤖 **Prédiction de Prix par IA :** Utilisation de modèles de Machine Learning entraînés pour suggérer le meilleur prix de vente selon les caractéristiques du produit (matériaux, taille, catégorie).
- 🔍 **Explicabilité (XAI) :** Visualisation de l'impact de chaque caractéristique sur le prix final pour garantir la transparence des résultats (via `SHAP`).
- 🎨 **Interface Moderne :** Une expérience utilisateur fluide, professionnelle et dynamique construite avec Next.js, Tailwind CSS et des animations GSAP.
- 🔒 **Sécurité :** Authentification robuste (JWT, hachage Argon2/Passlib) et gestion sécurisée des utilisateurs.

---

## 🛠️ Technologies Utilisées

### ⚙️ Backend (API & Machine Learning)
- **Langage & Framework :** Python, FastAPI
- **Base de données :** PostgreSQL (via `psycopg2`), SQLAlchemy (ORM)
- **Sécurité :** JWT (`python-jose`), `argon2-cffi`, `passlib`
- **Machine Learning :** Scikit-Learn, XGBoost, Pandas, SHAP
- **Web Scraping :** BeautifulSoup4, Requests
- **Visualisation :** Matplotlib, Seaborn

### 🎨 Frontend (Interface Utilisateur)
- **Framework :** Next.js 16 (React 19)
- **Langage :** TypeScript
- **Style :** Tailwind CSS
- **Animations :** GSAP (GreenSock Animation Platform)
- **Icônes :** Lucide React

---

## 📂 Structure du Projet

Le dépôt est divisé en deux grandes parties :

```tree
faircraft-ai/
├── faircraftAi-backend/      # API FastAPI et Pipeline d'Intelligence Artificielle
│   ├── src/
│   │   ├── api/              # Endpoints HTTP (ex: authentification, prédictions)
│   │   ├── ml/               # Pipeline Machine Learning (IA)
│   │   │   ├── etl_pipeline.py      # Nettoyage et préparation des données
│   │   │   ├── train_model.py       # Entraînement des modèles M.L.
│   │   │   ├── scrape_etsy.py       # Script de collecte sur Etsy
│   │   │   └── external_service.py  # Comparaison (Benchmark) avec une IA externe
│   │   └── app/              # Configuration, sécurité, modèles SQLAlchemy de la BDD
│   ├── data/                 # Données brutes et procesées (.csv, .xlsx)
│   └── requirements.txt      # Dépendances Python
│
└── faircraftAi-frontend/     # Application Web Front-end
    ├── src/
    │   ├── app/              # Dossiers des pages (Login, Register, Dashboard...)
    │   └── components/       # Composants React réutilisables (Boutons, Formulaires...)
    └── package.json          # Dépendances Node.js (npm)
```

---

## 💻 Installation et Lancement Rapide

### 1. Prérequis
- `Python` 3.9+ ou supérieur
- `Node.js` 18+ ou supérieur
- Une base de données `PostgreSQL` en cours d'exécution

### 2. Démarrer le Backend (L'API Web & IA)

```bash
# Obtenir l'accès au bout du backend
cd faircraftAi-backend

# Créer un environnement virtuel Python
python -m venv .venv

# Activer l'environnement virtuel (Sous Windows)
.venv\Scripts\activate
# (Ou sous macOS/Linux) : source .venv/bin/activate

# Installer l'ensemble des dépendances nécessaires
pip install -r requirements.txt

# Créer ou mettre à jour le fichier racine ".env"
# S'assurez de bien lier votre DATABASE_URL et de mettre un JWT_SECRET_KEY

# Lancer FastAPI en local
uvicorn src.main:app --reload
```
*Votre API prend vie sur : [http://127.0.0.1:8000](http://127.0.0.1:8000)*
*(Documentation automatique trouvable sur `http://127.0.0.1:8000/docs`)*

### 3. Démarrer le Frontend (L'Interface Visuelle)

```bash
# Revenir à la racine, puis naviguer dans le dossier frontend
cd ../faircraftAi-frontend

# Installer les dépendances NPM
npm install

# Démarrer le serveur Next.js en mode développeur
npm run dev
```
*Votre page sera accessible sur : [http://localhost:3000](http://localhost:3000)*

---

## 🧠 Le fonctionnement de l'IA (Machine Learning)

Le workflow complet du Machine Learning se traduit en 5 étapes principales :
1. **Extraction brute :** Exécution de `scrape_etsy.py` afin de récupérer les catalogues de produits cibles.
2. **Transformation de la donnée (ETL) :** `etl_pipeline.py` vient lisser les valeurs aberrantes (outliers), imputer les manquements et créer les encodages adéquats.
3. **Phase d'Entraînement :** `train_model.py` génère des modèles prédictifs (comme le `RandomForestRegressor` ou le `XGBoost`). En utilisant la validation croisée (`RandomizedSearchCV`), l'algorithme se perfectionne de lui-même.
4. **Évaluation & Analyse :** Des diagnostics via des graphiques `SHAP` et des métriques courantes (RMSE, MAE, R²) sont observés pour juger la précision des prédictions estimées.
5. **Prédiction finale :** Dès qu'un client rentre de nouvelles spécifications dans le service, le modèle renvoie le meilleur prix basé sur la connaissance ingérée.

---

> Ce projet fait partie de la formation interne Simplon.
