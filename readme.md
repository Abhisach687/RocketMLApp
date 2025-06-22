.venv\Scripts\activate

python -m src.clean # uses paths & options in config.yaml

# OR override defaults:

python -m src.clean --raw_dir data/raw --out_dir data/interim --cfg config.yaml
