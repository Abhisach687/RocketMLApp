# SmartRocket Analytics Dashboard

A clean, intuitive dashboard for analyzing 2015 e-commerce data and model performance. This application provides comprehensive insights into historical e-commerce data using trained machine learning models.

## 🚀 Features

- **Clean Data Analysis**: Interactive visualizations of 2015 e-commerce data
- **Model Performance**: Comprehensive analysis of trained ML models (LightGBM, GRU4Rec)
- **Business Insights**: Automated generation of business intelligence from data
- **Modern UI**: Clean, readable interface with perfect contrast and accessibility
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📊 Dashboard Sections

### 1. Data Overview

- Key metrics and KPIs
- Data sample preview
- Quick business insights
- Data quality indicators

### 2. Sales Analysis

- **Sales Trends Over Time**: Time series analysis with moving averages
- **Category Performance**: Comparative analysis across product categories
- **Daily Sales Patterns**: Weekly and hourly sales patterns
- **Sales Distribution**: Statistical distribution of sales data

### 3. Model Performance

- **Forecast Accuracy Metrics**: R², MAPE, RMSE, MAE
- **Visual Performance Analysis**: Forecast vs Actual scatter plots
- **Model Insights**: Automated performance interpretation
- **Model Status**: Real-time status of all available models

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "Rocket Singh"
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the data** (if not already done)

   ```bash
   # Clean raw data
   python src/clean.py

   # Generate features
   python src/features.py

   # Perform EDA (optional)
   python src/EDA.py
   ```

5. **Train models** (if not already done)

   ```bash
   # Train LightGBM models
   python src/forecast_lightgbm.py
   python src/tune_lightgbm.py

   # Train GRU4Rec models (optional)
   python src/GRU4REC_baseline.py
   python src/tune_GRU4REC.py
   ```

## 🎯 Running the Dashboard

### Start the Application

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

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
