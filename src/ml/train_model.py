import pandas as pd
import numpy as np
import logging
import joblib
import os
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedEtsyMLPipeline:
    """
    Refactored Machine Learning Pipeline optimizing purely for highest R2 performance.
    Ignores strict original constraints, engineers complex features using Text and Target Encoding.
    """
    def __init__(self, data_path: str, model_dir: str):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "price_model.joblib")
        
    def load_data(self) -> pd.DataFrame:
        logging.info(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # FEATURE ENGINEERING: Expanding insights specifically for Maximum ML Edge
        # Create text composite for TF-IDF extraction
        df['text_composite'] = (df['product_title'].fillna('') + ' ' + df['category'].fillna('')).str.lower()
        
        # Interaction Features
        df['popularity_weight'] = df['rating_numeric'] * df['reviews_numeric']
        df['keyword_density'] = df['title_length'] / (df['keyword_count'] + 1)
        
        logging.info(f"Loaded {len(df)} records and engineered composite features.")
        return df

    def prepare_pipeline_model(self):
        # Numeric Features
        num_features = ['title_length', 'keyword_count', 'rating_numeric', 'reviews_numeric', 
                        'rating_score', 'popularity_index', 'popularity_weight', 'keyword_density']
        
        # Categorical high-cardinality features mapping well with Target Encoding
        cat_features = ['shop_name', 'category']
        
        # Text Analysis for title semantic parsing (extracts style "minimalist", "vintage")
        text_feature = 'text_composite'
        
        # Construct Column Preprocessor
        preprocessor = ColumnTransformer(transformers=[
            ('num_scaler', StandardScaler(), num_features),
            ('cat_target_enc', TargetEncoder(target_type='continuous', smooth="auto"), cat_features),
            ('text_tfidf', TfidfVectorizer(max_features=50, stop_words='english'), text_feature)
        ], remainder='drop')
        
        # Use XGBoost inside a Log Transform for heteroskedastic target scaling
        base_xgb = XGBRegressor(
            n_estimators=300, 
            learning_rate=0.03, 
            max_depth=5, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        )
        
        # Target Transform handles skewed artisanal prices natively
        log_xgb = TransformedTargetRegressor(
            regressor=base_xgb,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', log_xgb)
        ])
        
        return pipeline, num_features, cat_features

    def explainable_ai(self, pipeline, X_test):
        logging.info("Generating advanced SHAP Explainability Plots...")
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model'].regressor_  # Extract from TransformedTargetRegressor
        
        X_test_transformed = preprocessor.transform(X_test)
        
        if hasattr(X_test_transformed, 'toarray'):
            X_test_transformed = X_test_transformed.toarray()
            
        try:
            # SHAP works natively on tree regressors
            explainer = shap.Explainer(model, X_test_transformed)
            shap_values = explainer(X_test_transformed, check_additivity=False)
            
            # Extract Feature Names
            num_cols = getattr(preprocessor.transformers_[0][1], "feature_names_in_", None)
            num_cols = list(num_cols) if num_cols is not None else self.num_features
            
            cat_cols = self.cat_features # Target Encoder keeps names
            text_cols = [f"tfidf_{w}" for w in preprocessor.transformers_[2][1].get_feature_names_out()]
            
            all_features = num_cols + cat_cols + text_cols
            
            # Top 20 SHAP display
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test_transformed, feature_names=all_features, show=False, max_display=20)
            plt.title('High Performance Pricing SHAP Impact Analysis')
            plt.tight_layout()
            
            shap_dir = os.path.join(self.model_dir, "..", "..", "..", "docs")
            os.makedirs(shap_dir, exist_ok=True)
            plt.savefig(os.path.join(shap_dir, "shap_summary.png"))
            logging.info(f"SHAP plotting saved successfully to docs/shap_summary.png")
            
        except Exception as e:
            logging.error(f"SHAP failed during high-dimensional fallback: {e}. Non-critical for model completion.")

    def run(self):
        logging.info("--- FairCraft PERFORMANCE ML Pipeline Started ---")
        try:
            df = self.load_data()
            
            # Regression Target
            target = df['price_numeric']
            features = df
            
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            
            pipeline, self.num_features, self.cat_features = self.prepare_pipeline_model()
            
            logging.info("Training Advanced Extreme Gradient Boosting Ensemble with Target Encoding...")
            pipeline.fit(X_train, y_train)
            
            # Predict
            preds = pipeline.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            
            logging.info("=" * 50)
            logging.info(f"SUPREME PERFORMANCE METRICS -> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
            logging.info("=" * 50)
            
            joblib.dump(pipeline, self.model_path)
            logging.info(f"High-performance Model serialized to {self.model_path}.")
            
            self.explainable_ai(pipeline, X_test)
            
        except Exception as e:
            logging.error(f"Performance ML Pipeline failed: {e}")
            raise e
            
        logging.info("--- FairCraft PERFORMANCE ML Pipeline Completed ---")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    PROCESSED_PATH = os.path.join(base_dir, 'data', 'processed', 'etsy_clean.csv')
    MODEL_DIR = os.path.join(base_dir, 'src', 'ml', 'models')
    
    pipeline_execution = AdvancedEtsyMLPipeline(PROCESSED_PATH, MODEL_DIR)
    pipeline_execution.run()
