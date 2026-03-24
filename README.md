# 🌟 FairCraft AI - Backend

**FairCraft AI** est une plateforme innovante conçue pour aider les artisans à estimer de manière juste et compétitive le prix de leurs créations faites à la main. En s'appuyant sur des données réelles du marché (Etsy) et des modèles de Machine Learning performants (**Random Forest**, **XGBoost**), l'application fournit des recommandations de prix intelligentes, explicables et exploitables.

---

## 🚀 Fonctionnalités Principales

- 📊 **Scraping & Analyse du Marché :** Collecte automatisée de produits artisanaux sur **Etsy** pour une base de données comparative constamment mise à jour.
- 🤖 **Prédiction de Prix par IA :** Modèles ML entraînés pour suggérer des prix de vente optimaux basés sur les matériaux, la catégorie, le volume de ventes et les avis.
- 📉 **Multi-scénarios & Simulations "What-if" :** Testez l'impact d'un changement de matériaux ou d'une montée en gamme sur votre rentabilité avant production.
- 🔍 **Explicabilité (XAI) :** Transparence totale grâce aux valeurs **SHAP**, montrant l'influence précise de chaque caractéristique sur le prix suggéré.
- 🔒 **Architecture Sécurisée :** API FastAPI robuste avec authentification JWT (Passlib, Argon2) et intégration ORM SQLAlchemy.
- 🏥 **Health Monitoring :** Logging structuré et gestion efficace des dépendances pour une disponibilité maximale.

---

## 🛠️ Technologies Utilisées

### ⚙️ Backend & Machine Learning
- **Framework :** Python, **FastAPI**
- **Machine Learning :** Scikit-Learn, **XGBoost**, Pandas, NumPy
- **Explicabilité :** **SHAP** (SHapley Additive exPlanations)
- **Base de données :** PostgreSQL, **SQLAlchemy** (ORM)
- **Sécurité :** **JWT** (`python-jose`), **Argon2-cffi**, `passlib`
- **Scraping :** BeautifulSoup4, Requests
- **Visualisation :** Matplotlib, Seaborn

---

## 📂 Structure du Projet (Backend)

Le backend est structuré de manière modulaire pour séparer la logique API du pipeline de données :

```tree
faircraftAi-backend/
├── src/
│   ├── app/                 # Noyau de l'API FastAPI
│   │   ├── core/            # Configuration globale et sécurité
│   │   ├── db/              # Configuration de la base de données (PostgreSQL)
│   │   ├── dependencies/    # Injection de dépendances (Modèles ML, DB, Auth)
│   │   ├── models/          # Modèles SQLAlchemy (User, Prediction, etc.)
│   │   ├── routers/         # Endpoints de l'API (Auth, Predictions)
│   │   ├── schemas/         # Modèles Pydantic pour la validation
│   │   ├── services/        # Logique métier (Calculs, appels aux modèles ML)
│   │   └── main.py          # Point d'entrée de l'application
│   ├── ml/                  # Pipeline Intelligence Artificielle
│   │   ├── etl_pipeline.py  # Nettoyage et transformation des données
│   │   ├── train_model.py   # Scripts d'entraînement des modèles
│   │   ├── scrape_etsy.py   # Collecte de données sur Etsy
│   │   ├── external_service.py # Benchmarking avec IA externes
│   │   └── models/          # Modèles entraînés sauvegardés (.pkl, .json)
│   └── utils/               # Fonctions utilitaires helper
├── data/                    # Données brutes et procesées (CSV/Parquet)
├── requirements.txt         # Dépendances Python
└── .env                     # Variables d'environnement
```

---

## 💻 Installation et Lancement

### 1. Prérequis
- `Python` 3.9+ 
- `PostgreSQL` en cours d'exécution

### 2. Configuration du Backend

```bash
# Entrer dans le dossier backend
cd faircraftAi-backend

# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.venv\Scripts\activate
# (Unix/macOS) : source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Variables d'Environnement
Créez un fichier `.env` à la racine de `faircraftAi-backend/` :
```env
DATABASE_URL=postgresql://user:password@localhost/faircraft_db
JWT_SECRET_KEY=votre_cle_secrete_ici
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### 4. Démarrage (Local)
```bash
# Lancer le serveur avec auto-reload
uvicorn src.app.main:app --reload
```

### 5. Démarrage (Docker)
Si vous préférez utiliser Docker pour lancer le backend et la base de données PostgreSQL simultanément :

```bash
# Lancer les services avec Docker Compose
docker-compose up --build
```

*L'API est accessible sur : [http://127.0.0.1:8000](http://127.0.0.1:8000)*  
*Documentation Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

---

## 🧠 Cycle de Vie de la Donnée

Le pipeline ML suit un flux rigoureux :
1. **Extraction :** Collecte via `scrape_etsy.py`.
2. **ETL :** Nettoyage des outliers et imputation des valeurs manquantes (`etl_pipeline.py`).
3. **Entraînement :** Recherche d'hyperparamètres via `RandomizedSearchCV` et persistance du meilleur modèle (`train_model.py`).
4. **Utilisation :** Le service de prédiction charge le modèle au démarrage via l'injection de dépendances FastAPI pour des réponses ultra-rapides.

---

> Ce projet a été développé dans le cadre de la formation **Simplon**.
