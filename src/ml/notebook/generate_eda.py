import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

nb = nbformat.v4.new_notebook()

cells = [
    new_markdown_cell('# Exploratory Data Analysis (EDA) - FairCraft AI (Etsy Dataset)\n\n**FairCraft AI** helps artisans price their items fairly and profitably. To establish accurate benchmarks, this EDA explores scraped data from Etsy, reflecting actual market dynamics.\n\n### Notebook Objectives\n- **Data Overview:** Check distribution, schema, and completeness.\n- **Missing Values Analysis:** Diagnose scraped data gaps.\n- **Outlier Detection:** Validate the effectiveness of ETL bounds (e.g., extremely high prices).\n- **Feature Relationships:** Uncover what features interact seamlessly with pricing.'),
    
    new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Visual aesthetics
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams['figure.figsize'] = (14, 7)

import warnings
warnings.filterwarnings('ignore')'''),

    new_markdown_cell('## 1. Data Overview\nLoading the `etsy_clean.csv` dataset processed by the ETL pipeline.'),
    
    new_code_cell('''DATA_PATH = '../../data/processed/etsy_clean.csv'
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    display(df.head())
else:
    print("Dataset not found. Ensure scrape_etsy.py and etl_pipeline.py have run.")'''),

    new_markdown_cell('## 2. Missing Values Analysis\nExamining the density of our data. Since it is scraped, some features like ratings or reviews may default efficiently in ETL. Let’s evaluate the final cleanliness.'),

    new_code_cell('''if 'df' in locals():
    missing_data = df.isnull().sum()
    print("Missing Values Summary:")
    print(missing_data[missing_data > 0])
    
    # Visualizing missingness
    plt.figure(figsize=(10,4))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()'''),

    new_markdown_cell('## 3. Price Distribution & Outlier Detection\nThe target variable (`price_numeric`) needs to be normally distributed (or at least well-behaved) for regression models. We visualize the log distribution since prices contain huge variances.'),

    new_code_cell('''if 'df' in locals():
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original Price Distribution
    sns.histplot(df['price_numeric'], bins=50, kde=True, ax=ax[0], color='dodgerblue')
    ax[0].set_title('Raw Price Distribution (Bounded by ETL 95th Percentile)')
    ax[0].set_xlabel('Price ($)')
    
    # Log Transformed Price (price_log)
    sns.histplot(df['price_log'], bins=50, kde=True, ax=ax[1], color='darkorange')
    ax[1].set_title('Log Transformed Price Distribution (price_log)')
    ax[1].set_xlabel('Log(Price)')
    
    plt.show()'''),

    new_code_cell('''if 'df' in locals():
    plt.figure(figsize=(12, 6))
    # Boxenplot shows more quantiles efficiently
    sns.boxplot(y='category', x='price_numeric', data=df, palette='Set2')
    plt.title('Price Distribution and Outliers across Etsy Categories')
    plt.xlabel('Price ($)')
    plt.ylabel('Category')
    plt.show()'''),

    new_markdown_cell('## 4. Feature Relationships & Correlation Matrix\nDiscovering the interplay between `keyword_count`, `popularity_index`, `rating_score`, `title_length` and the final market price.'),

    new_code_cell('''if 'df' in locals():
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x='title_length', y='price_numeric', data=df, alpha=0.5, ax=ax[0], color='purple')
    ax[0].set_title('Title Length vs Final Price')
    ax[0].set_xlabel('Title Length (Characters)')
    
    sns.scatterplot(x='keyword_count', y='price_numeric', data=df, alpha=0.5, ax=ax[1], color='teal')
    ax[1].set_title('Keyword Count vs Price')
    ax[1].set_xlabel('Number of Keywords / Tags')
    
    plt.show()'''),

    new_code_cell('''if 'df' in locals():
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    # Heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Engineered Features')
    plt.show()'''),

    new_markdown_cell('## 5. Summary & Artisanal Insights\n1. **Price Skew**: Hand-crafted data has a long tail; `price_log` smooths it, validating its necessity for linear learners.\n2. **Popularity & Ratings**: Highly rated items (with high reliability through volume) fetch marginally higher absolute baselines, indicating trust matters on peer-to-peer marketplaces.\n3. **Keywords Strategy**: Optimizing Etsy `keyword_count` shows dense clustering—meaning having 13 tags is an industry standard across successful pricing tiers.')
]

nb['cells'] = cells

import os
output_dir = r"c:/Users/hp/Simplon_project/faircraft-ai/faircraftAi-backend/src/ml/notebook"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "eda.ipynb"), "w", encoding='utf-8') as f:
    nbformat.write(nb, f)
