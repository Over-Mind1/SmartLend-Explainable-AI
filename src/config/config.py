from pathlib import Path
from dotenv import load_dotenv
import os
import yaml

# load .env
load_dotenv()

# ---------- BASE PATH ----------
BASE_PATH = Path(__file__).resolve().parents[2]

# ---------- LOAD YAML ----------
CONFIG_YAML_PATH = BASE_PATH / "src" / "config" / "config.yaml"

with open(CONFIG_YAML_PATH, "r") as file:
    config = yaml.safe_load(file)

# ---------- APP CONFIG ----------
APP_NAME = os.getenv("APP_NAME", "DefaultAppName")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
APP_KEY = os.getenv("APP_KEY", "your_default_key")
APP_ENV = os.getenv("APP_ENV", "development")

# ---------- DATA PATHS ----------
TRAIN_DATA_PATH = BASE_PATH / config["Data"]["train_path"]
X_TEST_DATA_PATH = BASE_PATH / config["Data"]["X_test_path"]
Y_TEST_DATA_PATH = BASE_PATH / config["Data"]["Y_test_path"]
# ---------- MODEL PATHS ----------
lgbm_save_path = BASE_PATH / config["Model"]["lgbm_save_path"]
lgbm_report_path = BASE_PATH / config["Model"]["lgbm_report_path"]
xgb_save_path = BASE_PATH / config["Model"]["xgb_save_path"]
# ---------- PROCESSOR PATH ----------
processor_path = BASE_PATH / config["Processor"]["Processor_path"]
lgbm_shap_path = BASE_PATH / config["lgbm_shap"]["shap_save_path"]
# ---------- LOG ----------
print("=" * 30)
print("Configurations loaded successfully")
print(f"APP_NAME: {APP_NAME}")
print(f"APP_VERSION: {APP_VERSION}")
print("=" * 30)
