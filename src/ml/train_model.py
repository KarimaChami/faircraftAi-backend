import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def perform_feature_engineering(df):
    """
    Apply advanced feature engineering to improve predictive power.
    """
    print("Performing Feature Engineering...")
    
    # 1. Derived Costs & Ratios
    # est_base_cost is our proxy for materials+labor
    # Let's add interaction features
    df['cost_per_hour'] = df['est_base_cost'] / (df['est_time_h'] + 1e-5)
    df['complexity_index'] = df['feature_count'] * df['est_time_h']
    df['visual_appeal'] = df['image_count'] * df['rating']
    
    # 2. Textual density (description detail per base cost)
    df['description_density'] = df['description_len'] / (df['est_base_cost'] + 1e-5)
    
    # 3. Log transformation for skewed numerical features (like price often is)
    # Note: We won't log target directly here to keep MAE/RMSE comparable unless needed,
    # but we can log transform skewed inputs
    df['log_description_len'] = np.log1p(df['description_len'])
    df['log_base_cost'] = np.log1p(df['est_base_cost'])
    
    return df

def treat_outliers(df, columns):
    """
    Cap outliers using IQR method to prevent model distortion.
    """
    print("Treating outliers...")
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def train_faircraft_models():
    # Load data
    DATA_PATH = 'data/processed/processed_artisans.csv'
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run pipeline first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Initial dataset size: {len(df)}")

    # --- Step 1: Preprocessing ---
    # Handle missing values (though ETL handles most)
    df['title'] = df['title'].fillna('Unknown')
    df['store'] = df['store'].fillna('Unknown')
    
    # Feature Engineering
    df = perform_feature_engineering(df)
    
    # Outlier treatment for key numerical columns
    cols_to_cap = ['price', 'est_base_cost', 'est_time_h', 'weight_oz']
    df = treat_outliers(df, cols_to_cap)
    
    # Encoding categorical variables
    le_cat1 = LabelEncoder()
    df['cat1_encoded'] = le_cat1.fit_transform(df['category_1'])
    
    # Define features and target
    features = [
        'rating', 'rating_count', 'feature_count', 'image_count', 
        'description_len', 'weight_oz', 'est_base_cost', 'est_time_h',
        'cat1_encoded', 'cost_per_hour', 'complexity_index', 'visual_appeal',
        'description_density', 'log_description_len', 'log_base_cost'
    ]
    target = 'price'
    
    X = df[features]
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling pipeline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 2: Model Tuning (Simplified for Demo, but robust) ---
    print("\nTuning Models with Cross-Validation...")
    
    # 1. Random Forest Tuning
    print("Optimizing RandomForest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best RF Params: {rf_grid.best_params_}")
    
    # 2. XGBoost Tuning
    print("Optimizing XGBoost...")
    xgb_params = {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 6],
        'subsample': [0.8, 1.0]
    }
    xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=3, scoring='r2', n_jobs=-1)
    xgb_grid.fit(X_train_scaled, y_train)
    best_xgb = xgb_grid.best_estimator_
    print(f"Best XGB Params: {xgb_grid.best_params_}")
    
    # --- Step 3: Evaluation ---
    def evaluate(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.4f}")
        return r2

    r2_rf = evaluate(best_rf, X_test_scaled, y_test, "Optimized RandomForest")
    r2_xgb = evaluate(best_xgb, X_test_scaled, y_test, "Optimized XGBoost")
    
    # Save the best model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    best_model = best_xgb if r2_xgb > r2_rf else best_rf
    joblib.dump(best_model, os.path.join(model_dir, 'faircraft_price_predictor.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le_cat1, os.path.join(model_dir, 'category_encoder.joblib'))
    
    # --- Step 4: Explainability (SHAP) ---
    print("\nGenerating Detailed Explainability Reports (SHAP)...")
    # Using a sample for SHAP calculation speed
    shap_sample = X_test_scaled[:200]
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(shap_sample)
    
    # Save SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test.iloc[:200], feature_names=features, show=False)
    plt.title("FairCraft AI: Feature Impact on Price")
    plt.tight_layout()
    plt.savefig('docs/shap_summary.png')
    print("SHAP Summary plot saved to docs/shap_summary.png")

    # --- Step 5: Recommendations & Scenarios ---
    sample_idx = 0
    sample_artisan_raw = X_test.iloc[[sample_idx]]
    sample_artisan_scaled = X_test_scaled[[sample_idx]]
    
    predicted_price = best_xgb.predict(sample_artisan_scaled)[0]
    
    print(f"\n--- Production Report for Sample Artisan ---")
    print(f"Predicted Recommended Price: ${predicted_price:.2f}")
    print(f"Minimum Price (Cost-Focused): ${sample_artisan_raw['est_base_cost'].values[0] * 1.2 :.2f}")
    print(f"Premium Price (Value-Focused): ${predicted_price * 1.4 :.2f}")
    
    print("\nAI Artisan Insight:")
    # Simple logic-based insights derived from feature importance
    top_feature_idx = np.argmax(np.abs(shap_values[sample_idx]))
    top_feature_name = features[top_feature_idx]
    print(f"Explanation: '{top_feature_name}' had the strongest impact on this specific recommendation.")

if __name__ == "__main__":
    train_faircraft_models()
