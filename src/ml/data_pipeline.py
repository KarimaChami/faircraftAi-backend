import os
import sys
import logging
import pandas as pd

# Add the root directory to sys.path to import etl_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from etl_pipeline import run_etl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FairCraft-Pipeline")

def validate_data(df):
    """Simple data validation rules."""
    logger.info("Starting Data Validation...")
    
    # Check for empty dataframe
    if df.empty:
        logger.error("Validation Failed: Dataframe is empty.")
        return False
        
    # Check for required columns
    required_cols = ['price', 'est_base_cost', 'est_time_h']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Validation Failed: Missing required column {col}")
            return False
            
    # Check for negative prices
    if (df['price'] < 0).any():
        logger.warning("Validation Warning: Found negative prices. Cleaning...")
        df = df[df['price'] >= 0]
        
    logger.info(f"Validation Passed: {len(df)} records ready for ML.")
    return True

def run_pipeline():
    RAW_DATA = 'data/raw/meta_Handmade_Products.jsonl'
    PROCESSED_DATA = 'data/processed/processed_artisans.csv'
    
    logger.info("--- FairCraft AI Data Pipeline Started ---")
    
    try:
        # 1. Ingestion & ETL
        logger.info(f"Step 1: Ingesting raw data from {RAW_DATA}")
        run_etl(RAW_DATA, PROCESSED_DATA, limit=50000)
        
        # 2. Loading for Validation
        logger.info(f"Step 2: Loading processed data for validation.")
        df = pd.read_csv(PROCESSED_DATA)
        
        # 3. Validation
        if validate_data(df):
            logger.info("Step 3: Validation Successful.")
        else:
            logger.error("Step 3: Validation Failed.")
            return
            
        logger.info("--- Pipeline Completed Successfully ---")
        
    except Exception as e:
        logger.error(f"Pipeline Crashed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
