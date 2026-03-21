import pandas as pd
import joblib
import os

# 🔹 Chemin vers le modèle entraîné
MODEL_PATH = r"C:\Users\hp\Simplon_project\faircraft-ai\faircraftAi-backend\src\ml\models\price_model.joblib"
model = joblib.load(MODEL_PATH)

# 🔹 Exemple d'un nouveau produit à prédire
# ⚠️ Les colonnes doivent correspondre à celles utilisées dans le pipeline
example_product = pd.DataFrame([{
    "product_title": "Minimalist Silver Ring",
    "category": "handmade jewelry",
    "shop_name": "SilverArts",
    "title_length": 21,
    "keyword_count": 3,
    "rating_numeric": 4.9,
    "reviews_numeric": 27,
    "rating_score": 4.85,
    "popularity_index": 132,
    "price_numeric": 0  # placeholder, non utilisé pour prédiction
}])

# ⚡️ Ajouter les features composites comme dans ton pipeline
example_product['text_composite'] = (example_product['product_title'].fillna('') + ' ' + example_product['category'].fillna('')).str.lower()
example_product['popularity_weight'] = example_product['rating_numeric'] * example_product['reviews_numeric']
example_product['keyword_density'] = example_product['title_length'] / (example_product['keyword_count'] + 1)

# 🔹 Prédiction
predicted_price = model.predict(example_product)
print(f"💰 Predicted Price: {predicted_price[0]:.2f} USD")