# Create the virtual environment

python -m venv .venv
.venv\Scripts\activate

# Install the required packages

pip install -r requirements.txt

# Run the scripts

python -m src.clean # uses paths & options in config.yaml

# OR override defaults:

python -m src.clean --raw_dir data/raw --out_dir data/interim --cfg config.yaml

# (3) Run the feature script

python -m src.features --cfg config.yaml

# 1. Feature engineering

python -m src.features

# 2. Baseline model

python -m src.forecast_lightgbm

# 3. Hyper‑parameter tuning

python -m src.tune_lightgbm
