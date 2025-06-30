# 🚀 SmartRocket Analytics Dashboard

A comprehensive ML-powered analytics dashboard for e-commerce data analysis, featuring advanced forecasting models (LightGBM) and recommendation systems (GRU4Rec). This application transforms raw retail data into actionable business insights through interactive visualizations and trained machine learning models.

## ✨ Key Features

- **🔮 Smart Forecasting**: Advanced LightGBM models with hyperparameter tuning
- **🛍️ AI Recommendations**: GRU4Rec-based session-aware recommendation engine
- **📊 Business Intelligence**: Automated insights generation and trend analysis
- **🎨 Modern Interface**: Clean, accessible UI with perfect contrast and responsive design
- **📈 Interactive Visualizations**: Real-time charts and analytics powered by Plotly
- **🏷️ Smart Product Mapping**: CSV-based product and category naming system

## 🛠️ Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Data Processing Pipeline

```bash
# Step 1: Clean raw data (uses paths & options in config.yaml)
python -m src.clean

# OR override defaults:
python -m src.clean --raw_dir data/raw --out_dir data/interim --cfg config.yaml

# Step 2: Feature engineering
python -m src.features --cfg config.yaml

# OR with default config:
python -m src.features
```

### 4. Model Training Pipeline

```bash
# Step 1: Baseline LightGBM model
python -m src.forecast_lightgbm

# Step 2: Hyperparameter tuning
python -m src.tune_lightgbm

# Step 3: Baseline GRU4Rec model (optional)
python -m src.GRU4REC_baseline

# Step 4: GRU4Rec tuning (optional)
python -m src.tune_GRU4REC
```

### 5. Launch Dashboard

```bash
streamlit run app.py
```

Access the dashboard at: `http://localhost:8501`

## 📋 Requirements

### Core Dependencies

```
streamlit==1.40.2          # Dashboard framework
pandas==2.3.0              # Data manipulation
numpy==1.26.4               # Numerical computing
plotly==5.18.0              # Interactive visualizations
lightgbm==3.3.5             # Gradient boosting models
torch==2.7.1                # Deep learning framework
scikit-learn==1.7.0         # ML utilities
pyarrow==20.0.0             # Parquet file support
PyYAML==6.0.2               # Configuration management
```

### Optimization & Analysis

```
optuna==4.4.0               # Hyperparameter optimization
matplotlib==3.10.3          # Static plotting
seaborn==0.13.2            # Statistical visualizations
tqdm==4.67.1               # Progress bars
joblib==1.5.1              # Model persistence
```

## 📁 Project Architecture

```
SmartRocket Analytics/
├── 🎯 Core Application
│   ├── app.py                          # Main Streamlit dashboard
│   ├── config.yaml                     # Configuration settings
│   ├── product_mapping.csv             # Product/category name mappings
│   └── requirements.txt                # Python dependencies
│
├── 🔧 Source Code
│   └── src/
│       ├── clean.py                    # Data cleaning pipeline
│       ├── features.py                 # Feature engineering
│       ├── EDA.py                      # Exploratory data analysis
│       ├── forecast_lightgbm.py        # LightGBM baseline training
│       ├── tune_lightgbm.py           # LightGBM hyperparameter tuning
│       ├── GRU4REC_baseline.py        # GRU4Rec baseline model
│       └── tune_GRU4REC.py            # GRU4Rec optimization
│
├── 📊 Data Pipeline
│   └── data/
│       ├── raw/                        # Original CSV files
│       │   ├── events.csv
│       │   ├── item_properties_part1.csv
│       │   ├── item_properties_part2.csv
│       │   └── category_tree.csv
│       ├── interim/                    # Cleaned data
│       │   ├── events_clean.parquet
│       │   ├── item_properties.parquet
│       │   └── category_tree.parquet
│       └── processed/                  # Feature-engineered data
│           ├── forecast_features.parquet
│           └── reco_sequences.parquet
│
├── 🤖 Trained Models
│   └── artefacts/
│       ├── lightgbm_weighted.pkl       # Baseline LightGBM
│       ├── lightgbm_tuned_weighted.pkl # Optimized LightGBM
│       ├── gru4rec_baseline.pt         # Baseline GRU4Rec
│       ├── gru4rec_tuned.pt           # Optimized GRU4Rec
│       └── item2idx.json              # Item index mappings
│
└── 📈 Analysis Reports
    └── reports/
        ├── metrics_forecast_final.md   # Model performance metrics
        ├── metrics_forecast_tuned.md   # Tuned model results
        ├── lightgbm_feature_importance.png
        ├── lightgbm_prediction_quality.png
        └── eda/                        # Exploratory analysis outputs
```

## 🎛️ Dashboard Features

### 📊 Business Intelligence Tab

- **Key Performance Metrics**: Revenue, transactions, active products
- **Sales Trend Analysis**: Time series with moving averages
- **Category Performance**: Comparative revenue analysis
- **Automated Insights**: AI-generated business recommendations

### 🔮 Smart Forecasting Tab

- **Model Performance Metrics**: R², MAPE, RMSE, MAE
- **Forecast vs Actual Visualization**: Interactive scatter plots
- **Prediction Quality Analysis**: Model accuracy assessment
- **Future Revenue Projections**: 7-day ahead forecasting

### 🛍️ AI Recommendations Tab

- **Session-Based Recommendations**: GRU4Rec powered suggestions
- **User Behavior Analysis**: Session length and interaction patterns
- **Trending Products**: Most popular items across sessions
- **Interactive Session Explorer**: Deep dive into user journeys

### 🔍 Individual Analysis Tab

- **Product Deep Dive**: Individual item performance analysis
- **Category Analysis**: Comprehensive category-level insights
- **Sales Forecasting**: Item and category-specific predictions
- **Cross-Product Recommendations**: AI-powered product suggestions

## ⚙️ Configuration

### Main Configuration (`config.yaml`)

```yaml
# Application Settings
app:
  forecast_features_path: data/processed/forecast_features.parquet
  reco_sequences_path: data/processed/reco_sequences.parquet
  lightgbm_model_path: artefacts/lightgbm_weighted.pkl
  gru_model_path: artefacts/gru4rec.pt

# Data Processing
clean:
  raw_dir: data/raw
  out_dir: data/interim
  bot_threshold_per_day: 10000

features:
  forecast_horizon_days: 7
  rolling_window_days: 30
  min_interactions_per_user: 5

# Model Training
models:
  forecast:
    objective: regression
    metric: mae
    num_boost_round: 300
    early_stopping_rounds: 30
  reco:
    batch_size: 128
    embedding_dim: 32
    hidden_size: 64
    epochs: 5
```

### Product Mapping (`product_mapping.csv`)

The dashboard uses a CSV-based mapping system for meaningful product and category names:

```csv
category_id,category_name,item_id,item_name
1,Electronics & Tech,1006,iPhone 15 Pro Max
1,Electronics & Tech,1013,Samsung Galaxy S24 Ultra
2,Fashion & Apparel,2001,Nike Air Max 270
```

## 🔄 Data Processing Workflow

### Stage 1: Data Cleaning (`src/clean.py`)

**Input**: Raw CSV files from `data/raw/`

- `events.csv` - User interaction events
- `item_properties_part1.csv` - Product metadata
- `item_properties_part2.csv` - Additional product data
- `category_tree.csv` - Category hierarchy

**Process**:

- Parse timestamps (ms → UTC datetime)
- Whitelist valid event types
- Extract numeric values
- Remove missing critical fields
- Deduplicate records
- Cast IDs to appropriate types

**Output**: Cleaned Parquet files in `data/interim/`

### Stage 2: Feature Engineering (`src/features.py`)

**Forecasting Features**:

- Rolling sales aggregations (7, 14, 30 days)
- Lag features (1, 3, 7 days)
- Category-level features
- Trend and seasonality indicators

**Recommendation Features**:

- User session sequences
- Item interaction patterns
- Session length and frequency
- Item co-occurrence matrices

**Output**: Feature-engineered data in `data/processed/`

### Stage 3: Model Training

**LightGBM Forecasting**:

```bash
# Baseline model with weighted loss
python -m src.forecast_lightgbm
```

**Hyperparameter Optimization**:

```bash
# Optuna-based tuning (25 trials, 20min timeout)
python -m src.tune_lightgbm
```

**GRU4Rec Recommendations**:

```bash
# Session-based recommendation model
python -m src.GRU4REC_baseline
python -m src.tune_GRU4REC
```

## 📊 Model Performance

### LightGBM Forecasting Metrics

- **Baseline Model**: MAPE ~15-20%, R² ~0.75-0.85
- **Tuned Model**: MAPE ~12-18%, R² ~0.80-0.90
- **Features**: 20+ engineered features including rolling aggregations
- **Objective**: Poisson regression with weighted loss

### GRU4Rec Recommendation Metrics

- **Architecture**: LSTM-based session modeling
- **Embedding**: 32-dimensional item embeddings
- **Accuracy**: Top-5 recommendation hit rate ~25-35%
- **Features**: Session sequences with temporal dynamics

## 🎨 UI/UX Features

### Design System

- **Modern Aesthetic**: Clean lines, subtle shadows, professional gradients
- **Perfect Accessibility**: WCAG AA compliant contrast ratios
- **Responsive Layout**: Desktop and mobile optimized
- **Interactive Elements**: Hover effects, smooth transitions

### Data Visualizations

- **Plotly Integration**: Interactive charts with zoom, pan, hover
- **Color Coding**: Consistent brand colors (Rocket Red #dc2626)
- **Chart Types**: Line, bar, scatter, histogram, heatmap
- **Export Ready**: High-resolution chart downloads

## 🔧 Advanced Usage

### Custom Data Sources

To use your own data, ensure the following structure:

**Events Data**:

```python
columns = ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']
```

**Item Properties**:

```python
columns = ['itemid', 'property', 'value', 'categoryid']
```

### Model Customization

**Modify LightGBM Parameters**:

```python
# Edit config.yaml
models:
  forecast:
    num_leaves: 246
    learning_rate: 0.289
    bagging_fraction: 0.822
```

**GRU4Rec Architecture**:

```python
# Edit config.yaml
models:
  reco:
    hidden_size: 64
    embedding_dim: 32
    batch_size: 128
```

### Extending the Dashboard

**Add New Visualizations**:

1. Create chart function in `app.py`
2. Add to chart type selectbox
3. Follow Plotly styling conventions

**Integrate New Models**:

1. Update `load_models()` function
2. Add model evaluation logic
3. Include in performance metrics

## 🚨 Troubleshooting

### Common Issues

**"No data available" Error**:

- ✅ Run data pipeline: `python -m src.clean && python -m src.features`
- ✅ Check file paths in `config.yaml`
- ✅ Verify Parquet files exist in `data/processed/`

**"Model not found" Warning**:

- ✅ Train models: `python -m src.forecast_lightgbm`
- ✅ Check `artefacts/` directory for `.pkl` files
- ✅ Verify model paths in configuration

**Performance Issues**:

- ✅ Reduce date range filters
- ✅ Limit category selections
- ✅ Clear Streamlit cache: `streamlit cache clear`

**Import Errors**:

- ✅ Activate virtual environment
- ✅ Install requirements: `pip install -r requirements.txt`
- ✅ Check Python version (3.8+ required)

### Data Validation

**Check Data Integrity**:

```bash
# Validate cleaned data
python -c "import pandas as pd; print(pd.read_parquet('data/interim/events_clean.parquet').info())"

# Check feature data
python -c "import pandas as pd; print(pd.read_parquet('data/processed/forecast_features.parquet').shape)"
```

## 🚀 Deployment

### Local Development

```bash
streamlit run app.py --server.port 8501
```

### Production Deployment

```bash
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📈 Performance Optimization

### Data Loading

- Parquet format for fast I/O
- Streamlit caching for model loading
- Lazy loading of large datasets

### Model Optimization

- LightGBM early stopping
- Optuna hyperparameter tuning
- GPU acceleration support (PyTorch)

### UI Performance

- Efficient chart rendering
- Progressive data loading
- Optimized CSS and JavaScript

## 🔒 Security Considerations

- No external API calls
- Local data processing only
- Secure file path handling
- Input validation and sanitization

## 🤝 Contributing

1. **Fork Repository**: Create your own copy
2. **Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push Branch**: `git push origin feature/amazing-feature`
5. **Pull Request**: Submit for review

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LightGBM**: Microsoft's gradient boosting framework
- **GRU4Rec**: Session-based recommendation research
- **Streamlit**: Rapid web app development
- **Plotly**: Interactive visualization library
- **RetailRocket**: Dataset provider

---

**🚀 SmartRocket Analytics Dashboard** - Transforming E-commerce Data into Actionable Intelligence

_Ready to launch your data-driven insights to the next level!_ 📊✨

### Configuration

The application uses `config.yaml` for configuration. Key settings:

```yaml
app:
  forecast_features_path: "data/processed/forecast_features.parquet"
  reco_sequences_path: "data/processed/reco_sequences.parquet"
```

## 📁 Project Structure

```
Rocket Singh/
├── app.py                          # Main Streamlit application
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── assets/                         # Static assets
│   ├── clean_theme.css            # Modern CSS theme
│   └── category_lookup.csv        # Category name mappings
│
├── src/                           # Source code
│   ├── clean.py                   # Data cleaning pipeline
│   ├── features.py                # Feature engineering
│   ├── EDA.py                     # Exploratory data analysis
│   ├── forecast_lightgbm.py       # LightGBM training
│   ├── tune_lightgbm.py          # LightGBM hyperparameter tuning
│   ├── GRU4REC_baseline.py       # GRU4Rec baseline model
│   └── tune_GRU4REC.py           # GRU4Rec tuning
│
├── data/                          # Data directories
│   ├── raw/                       # Original data files
│   ├── interim/                   # Intermediate processed data
│   └── processed/                 # Final processed data
│
├── artefacts/                     # Trained models
│   ├── lightgbm_weighted.pkl     # Baseline LightGBM model
│   ├── lightgbm_tuned_weighted.pkl # Tuned LightGBM model
│   ├── gru4rec_baseline.pt       # Baseline GRU4Rec model
│   ├── gru4rec_tuned.pt          # Tuned GRU4Rec model
│   └── item2idx.json             # Item index mapping
│
└── reports/                       # Analysis reports and figures
    ├── metrics_forecast_final.md
    ├── metrics_forecast_tuned.md
    └── eda/                       # EDA outputs
```

## 📈 Data Pipeline

### 1. Data Cleaning

```bash
python src/clean.py
```

- Processes raw CSV files
- Handles missing values and data quality issues
- Outputs cleaned data to `data/interim/`

### 2. Feature Engineering

```bash
python src/features.py
```

- Creates forecasting features
- Generates recommendation sequences
- Outputs processed features to `data/processed/`

### 3. Model Training

#### LightGBM Models

```bash
# Train baseline model
python src/forecast_lightgbm.py

# Train tuned model with hyperparameter optimization
python src/tune_lightgbm.py
```

#### GRU4Rec Models (Optional)

```bash
# Train baseline GRU4Rec model
python src/GRU4REC_baseline.py

# Train tuned GRU4Rec model
python src/tune_GRU4REC.py
```

## 🎛️ Using the Dashboard

### Filters and Controls

**Sidebar Filters:**

- **Date Range**: Select specific time periods for analysis
- **Categories**: Filter by product categories
- **Model Status**: View which models are currently loaded

### Navigation

**Data Overview Tab:**

- View key metrics (total records, sales, items, categories)
- Examine data samples
- Read automated business insights

**Sales Analysis Tab:**

- Choose from multiple chart types
- Interactive visualizations with hover details
- Export-ready charts

**Model Performance Tab:**

- View forecast accuracy metrics
- Analyze prediction quality
- Compare model performance

### Key Metrics Explained

- **R² Score**: Coefficient of determination (0-1, higher is better)
- **MAPE**: Mean Absolute Percentage Error (%, lower is better)
- **RMSE**: Root Mean Square Error (currency units, lower is better)
- **MAE**: Mean Absolute Error (currency units, lower is better)

## 🔧 Troubleshooting

### Common Issues

**1. "No data available" message**

- Check that data files exist in `data/processed/`
- Run the data pipeline: `python src/clean.py && python src/features.py`
- Verify date filters are not too restrictive

**2. "No models loaded" error**

- Train models using the commands in the setup section
- Check that model files exist in `artefacts/`
- Verify file paths in `config.yaml`

**3. Import errors**

- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**4. Poor performance**

- Reduce date range for large datasets
- Limit category selection
- Close other resource-intensive applications

### Data Requirements

The dashboard expects the following data structure:

**Forecast Features** (`data/processed/forecast_features.parquet`):

- `date`: Date column (datetime)
- `itemid`: Item identifier
- `categoryid`: Category identifier
- `sales`: Actual sales values
- `forecast`: Model predictions

**Sequences** (`data/processed/reco_sequences.parquet`):

- User session data for recommendations
- Item interaction sequences

### Environment Variables

You can set the following environment variables to customize behavior:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
```

## 🎨 Customization

### Theming

The dashboard uses a modern CSS theme located in `assets/clean_theme.css`. You can customize:

- Colors and gradients
- Typography
- Layout spacing
- Component styling

### Adding New Visualizations

To add new chart types:

1. Add the chart option to the selectbox in the Sales Analysis tab
2. Implement the chart logic using Plotly
3. Follow the existing pattern for consistent styling

### Model Integration

To add new model types:

1. Update the `load_models()` function
2. Add model status indicators
3. Implement prediction and evaluation logic

## 📝 Data Sources

This dashboard analyzes historical e-commerce data from 2015, including:

- Product catalog information
- Customer interaction events
- Sales transactions
- Category hierarchies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the project documentation
3. Open an issue on GitHub

---

**SmartRocket Analytics Dashboard** - Transforming e-commerce data into actionable insights 📊🚀
