# FairCraft AI: ML Pipeline Documentation

## 1. Overview
The FairCraft AI machine learning pipeline is designed to provide artisans with fair and competitive pricing recommendations for their handmade products. The pipeline transforms raw marketplace data into actionable insights using a regression-based approach.

## 2. Data Flow
- **Raw Data Ingestion:** Extracts JSONL product metadata from `data/raw/`.
- **Transformation (ETL):** Cleans data and engineering features such as:
  - `est_base_cost`: Calculated based on description complexity and material proxies.
  - `est_time_h`: Estimated production hours.
  - `quality_score`: Derived from ratings and image counts.
- **Validation:** Ensures data integrity (e.g., non-negative prices, required features).
- **Processing:** Encodes categorical features and prepares CSV for ML in `data/processed/`.

## 3. Machine Learning Models
- **Random Forest Regressor:** Used for baseline establishment and feature importance.
- **XGBoost (Extreme Gradient Boosting):** Optimized for predictive accuracy and handling of non-linear artisanal pricing factors.
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE): Measures average prediction error in dollars.
  - RMSE: Penalizes larger outliers.
  - R²: Shows the proportion of variance explained by features.

## 4. Explainable AI (XAI)
- Uses **SHAP (SHapley Additive exPlanations)** to explain WHY a price was suggested.
- Human-readable outputs include: "Production time and material quality explain most of the price increase."

## 5. External Integrations
- Supports **OpenAI-based Advisors** to provide marketing optimizations (e.g., phrasing descriptions to justify premium pricing).

## 6. Business Logic - Pricing Scenarios
1. **Minimum Price:** 1.2x estimated base cost (ensure survival).
2. **Recommended Price:** Based on ML market analysis.
3. **Premium Price:** 1.3x recommended price (if quality score permits).
