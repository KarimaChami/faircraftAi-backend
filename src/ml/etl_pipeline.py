import json
import pandas as pd
import os
import glob
from tqdm import tqdm
import re

def extract_features(data):
    """
    Extract relevant features from the raw JSON record.
    """
    # Basic target: Price
    price = data.get('price')
    if price is None:
        return None
    
    try:
        price = float(price)
    except (ValueError, TypeError):
        return None

    # Title & Description Length (proxy for effort/detail)
    title = data.get('title', '')
    description = " ".join(data.get('description', []))
    
    # Categories (depth and main category)
    categories = data.get('categories', [])
    category_lvl_1 = categories[1] if len(categories) > 1 else 'Other'
    category_lvl_2 = categories[2] if len(categories) > 2 else 'Other'
    
    # Store/Brand
    store = data.get('store', 'Unknown')
    
    # Ratings (Market Perception)
    avg_rating = data.get('average_rating', 0)
    rating_count = data.get('rating_number', 0)
    
    # Features List (count features as a proxy for complexity)
    features = data.get('features', [])
    feature_count = len(features)
    
    # Images (more images often correlates with higher perceived value/professionalism)
    images = data.get('images', [])
    image_count = len(images)
    
    # Extract dimensions/weight if available in details
    details = data.get('details', {})
    weight = 0
    if 'Package Dimensions' in details:
        dim_str = details['Package Dimensions']
        # Try to extract weight (usually ends in Ounces or Pounds)
        weight_match = re.search(r'(\d+\.?\d*)\s*(Ounces|Pounds|g|kg)', dim_str)
        if weight_match:
            val = float(weight_match.group(1))
            unit = weight_match.group(2).lower()
            if unit == 'pounds': weight = val * 16
            elif unit == 'kg': weight = val * 35.274
            elif unit == 'g': weight = val * 0.035274
            else: weight = val # default Ounces
            
    # Simulate production cost and time if not present (using proxy metrics)
    # In a real scenario, this would be provided, but here we derive it 
    # to demonstrate the FairCraft AI business logic.
    estimated_base_cost = (len(description) / 100.0) + (feature_count * 2.0) + (weight * 0.5)
    estimated_time_hours = (len(description) / 500.0) + (feature_count * 0.5) + 1.0
    
    return {
        'title': title,
        'price': price,
        'category_1': category_lvl_1,
        'category_2': category_lvl_2,
        'store': store,
        'rating': avg_rating,
        'rating_count': rating_count,
        'feature_count': feature_count,
        'image_count': image_count,
        'description_len': len(description),
        'weight_oz': weight,
        'est_base_cost': round(estimated_base_cost, 2),
        'est_time_h': round(estimated_time_hours, 2),
        'quality_score': (avg_rating * 0.7) + (min(image_count, 5) * 0.3)
    }

def run_etl(input_path, output_path, limit=50000):
    print(f"Starting ETL: {input_path} -> {output_path}")
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    extracted_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in tqdm(f, desc="Processing JSONL"):
            try:
                data = json.loads(line)
                processed = extract_features(data)
                if processed:
                    extracted_data.append(processed)
                    count += 1
                
                if count >= limit:
                    break
            except Exception as e:
                continue
                
    df = pd.DataFrame(extracted_data)
    
    # Simple Cleaning
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    
    # Save as CSV for easier consumption in Notebook
    df.to_csv(output_path, index=False)
    print(f"ETL Complete. Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    RAW_DATA = 'data/raw/meta_Handmade_Products.jsonl'
    PROCESSED_DATA = 'data/processed/processed_artisans.csv'
    
    # We take a sample for the project demonstration
    run_etl(RAW_DATA, PROCESSED_DATA, limit=20000)
