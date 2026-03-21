import logging
import os
import subprocess
import time
import sys
# Configure logging system for the Automated Pipeline
LOG_PATH = os.path.join(os.path.dirname(__file__), 'data_pipeline.log')
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

class AutomatedDataPipeline:
    """
    Automated Data Pipeline for FairCraft AI (Etsy Dataset).
    Simulates a production-ready execution matching an Airflow DAG.
    Orchestrates: Ingestion (Scraping) -> Cleaning/ETL -> Validation -> Storage.
    """
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.scrape_script = os.path.join(self.base_dir, 'scrape_etsy.py')
        self.etl_script = os.path.join(self.base_dir, 'etl_pipeline.py')

    def run_stage(self, stage_name: str, script_path: str):
        logging.info(f"==> Starting Stage: {stage_name}")
        start_time = time.time()
        try:
            # Execute script as a subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.time() - start_time
            logging.info(f"==> Stage '{stage_name}' completed successfully in {elapsed:.2f} seconds.")
            # Logging subprocess output to the pipeline log implicitly
            for line in result.stdout.splitlines():
                if "INFO" in line or "WARNING" in line:
                    logging.info(f"[{stage_name} Out] {line.strip()}")
        except subprocess.CalledProcessError as e:
            logging.error(f"==> Stage '{stage_name}' FAILED with exit code {e.returncode}.")
            logging.error(f"Error output: {e.stderr}")
            raise Exception(f"Pipeline failed at stage {stage_name}")

    def validate_and_store(self):
        logging.info("==> Starting Validation & Storage Checks")
        processed_path = os.path.join(self.base_dir, '../../data/processed/etsy_clean.csv')
        if not os.path.exists(processed_path):
            logging.error("Validation Failed: Processed file 'etsy_clean.csv' does not exist.")
            raise FileNotFoundError(processed_path)
        
        # Simple file stats validation to ensure storage succeeded
        file_size = os.path.getsize(processed_path)
        if file_size < 100:  # Arbitrary threshold to check if file is basically empty
            logging.warning("Validation Warning: Processed file is suspiciously small.")
        else:
            logging.info(f"Validation Success: 'etsy_clean.csv' is ready for modeling (Size: {file_size} bytes).")

    def run_pipeline(self):
        logging.info("="*50)
        logging.info(f"FAIRCRAFT AI - DATA PIPELINE TRIGGERED AT {time.ctime()}")
        logging.info("="*50)
        
        try:
            # Task 1: Data Ingestion (Scraping Etsy)
            self.run_stage("Data Ingestion", self.scrape_script)
            
            # Task 2: Data Cleaning & Feature Engineering
            self.run_stage("Data Cleaning/ETL", self.etl_script)
            
            # Task 3: Validation & Storage Verification
            self.validate_and_store()
            
            logging.info("="*50)
            logging.info("PIPELINE COMPLETED SUCCESSFULLY. Ready for ML.")
            logging.info("="*50)
        except Exception as e:
            logging.error("="*50)
            logging.error(f"PIPELINE TERMINATED FATALLY: {e}")
            logging.error("="*50)

if __name__ == "__main__":
    pipeline = AutomatedDataPipeline()
    pipeline.run_pipeline()
