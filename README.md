# 🚀 Ultimate Bitcoin Price Forecaster

A comprehensive, state-of-the-art Bitcoin price prediction system that combines multiple machine learning models, technical analysis, sentiment data, and macroeconomic indicators to forecast Bitcoin prices with confidence intervals.

![Bitcoin Forecasting](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

### 🤖 Multiple ML Models
- **XGBoost**: Gradient boosting for robust predictions
- **LSTM**: Bidirectional LSTM with attention mechanism for sequence modeling
- **Prophet**: Facebook's time series forecasting optimized for Bitcoin cycles
- **CatBoost**: Gradient boosting with categorical feature handling
- **Ensemble**: Weighted combination of all models
- **Regime-Specific Models**: Different models for bull/bear/accumulation phases
- **Quantile Regression**: Confidence intervals (10th, 50th, 90th percentiles)
- **Directional Classification**: Predict up/down movements

### 📊 Data Sources
- **Price Data**: TwelveData API (OHLCV)
- **Volume**: Binance API for accurate volume data
- **Sentiment**: Fear & Greed Index from Alternative.me
- **Macro Data**: FRED API (Treasury yields, CPI, economic indicators)
- **On-Chain**: Blockchain.com (hash rate, difficulty, transactions)
- **Technical**: 50+ indicators via pandas_ta

### 🔧 Advanced Features
- **Hyperparameter Optimization**: Optuna for automated tuning
- **Time Series Cross-Validation**: Proper validation for time series data
- **Market Regime Detection**: Bull/bear/accumulation phase identification
- **Bitcoin Halving Cycles**: Specialized features for 4-year cycles
- **GPU Acceleration**: TensorFlow GPU support
- **Model Persistence**: Save/load trained models
- **Prediction Stability**: Multi-run validation
- **Comprehensive Logging**: Full audit trail

## 📋 Requirements

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow xgboost prophet optuna catboost lightgbm pandas-ta requests
```

### 🔑 API Keys Required
- **TwelveData**: For Bitcoin price data (free tier available)
- **FRED**: For macroeconomic data (free)

## 🚀 Quick Start

### Option 1: Web Dashboard (Recommended)
1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Launch Dashboard**:
```bash
python launch_dashboard.py
```

3. **Use the Web Interface**:
   - Enter your API keys in the sidebar
   - Configure parameters
   - Load data, train models, and generate predictions
   - View interactive charts and analysis

### Option 2: Python Script
1. **Get API Keys**:
   - TwelveData: [Register here](https://twelvedata.com/)
   - FRED: [Register here](https://fred.stlouisfed.org/docs/api/api_key.html)

2. **Run the System**:
```python
from bitcoinML7 import UltimateBitcoinForecaster

# Initialize
btc = UltimateBitcoinForecaster()

# Check GPU availability
btc.check_gpu()

# Set API keys
btc.fred_api_key = 'your_fred_api_key'

# Fetch data (up to 5000 days)
btc.fetch_bitcoin_data(days=2000, api_key='your_twelvedata_key')
btc.fetch_volume_from_binance(days=2000)

# Add features and train models
if btc.data is not None:
    btc.add_cycle_features()
    btc.calculate_indicators()
    btc.engineer_directional_features()
    btc.detect_market_regimes()
    
    # Train all models with hyperparameter optimization
    ml_data = btc.prepare_ml_data(lookback=30)
    if ml_data:
        X_scaled, y_scaled, dates, _ = ml_data
        btc.optimize_hyperparameters(X_scaled, y_scaled, dates, lookback=30)
        btc.train_all_models(test_size=0.1, lookback=30)
        
        # Generate future predictions with confidence intervals
        predictions = btc.predict_with_cycle_awareness(days_ahead=30)
        print(predictions)
```

### Option 3: Configuration File
1. **Edit config.yaml**:
```yaml
api:
  twelvedata_key: "your_api_key_here"
  fred_key: "your_fred_key_here"

models:
  lookback_days: 30
  test_size: 0.1

prediction:
  days_ahead: 30
```

2. **Run with Configuration**:
```python
import yaml
from bitcoinML7 import UltimateBitcoinForecaster

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with config
btc = UltimateBitcoinForecaster()
btc.fred_api_key = config['api']['fred_key']

# Use config parameters
btc.fetch_bitcoin_data(
    days=config['data']['days'], 
    api_key=config['api']['twelvedata_key']
)
```

## 📈 Model Performance

The system provides multiple evaluation metrics:
- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAPE**: Mean Absolute Percentage Error 
- **Directional Accuracy**: Percentage of correct up/down predictions
- **Cross-Validation**: Time series split validation scores

## 🔍 Understanding Predictions

### Output Format
```python
Date,Predicted_Returns,Predicted_Price,Lower_Price,Upper_Price
2024-01-01,0.0234,45678.90,43210.50,48123.45
```

### Confidence Intervals
- **Lower_Price**: 10th percentile (bearish scenario)
- **Predicted_Price**: 50th percentile (most likely scenario)  
- **Upper_Price**: 90th percentile (bullish scenario)

## 🛠️ Architecture

### Data Pipeline
```
Market Data → Feature Engineering → Technical Indicators → ML Models → Ensemble → Predictions
```

### Models Used
1. **XGBoost**: Tree-based gradient boosting
2. **LSTM**: Deep learning for sequence patterns
3. **Prophet**: Time series with seasonality and trends
4. **CatBoost**: Gradient boosting with automatic categorical handling
5. **Quantile Models**: Uncertainty quantification

### Feature Categories
- **Price Features**: OHLC, returns, volatility
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Volume Features**: OBV, CMF, VWAP, Force Index
- **Sentiment**: Fear & Greed Index
- **Macro**: Interest rates, inflation, economic indicators
- **On-Chain**: Hash rate, difficulty, transaction volume
- **Cycle Features**: Halving cycles, seasonal patterns

## ⚠️ Risk Disclaimer

**IMPORTANT**: This system is for educational and research purposes only. Cryptocurrency markets are extremely volatile and unpredictable. Key risks include:

- High volatility can cause rapid, significant losses
- Models may fail during unprecedented market conditions
- Past performance does not guarantee future results
- Regulatory changes can impact cryptocurrency values
- Technical analysis has inherent limitations

**DO NOT** use this for actual trading without:
- Thorough backtesting
- Risk management strategies
- Professional financial advice
- Understanding of cryptocurrency risks

## 📊 Example Output

```
--- Model Performance on Test Set ---
RMSE: 0.0234
MAPE: 0.0456  
Directional Accuracy: 0.6789

--- Future Predictions ---
         Date  Predicted_Returns  Predicted_Price  Lower_Price  Upper_Price
0  2024-01-01           0.0123        45678.90     43210.50     48123.45
1  2024-01-02          -0.0045        45473.12     42987.23     47965.78
...
```

## 🔧 Configuration

### Model Parameters
- `lookback`: Number of days to look back (default: 30)
- `test_size`: Fraction for test set (default: 0.1)
- `days_ahead`: Prediction horizon (default: 30)

### Feature Selection
- Automatic feature selection using RFECV
- Correlation analysis and redundancy removal
- Regime-specific feature importance

## 📁 File Structure

```
bitcoin-ml-forecaster/
├── bitcoinML7.py                # Main forecasting system
├── streamlit_dashboard.py       # Web dashboard interface
├── backtesting_framework.py     # Backtesting and strategy evaluation
├── visualization_utils.py       # Enhanced visualization tools
├── launch_dashboard.py          # Dashboard launcher script
├── config.yaml                  # Configuration file
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── models/                      # Saved model files
│   ├── xgb_model.pkl
│   ├── lstm_model.h5
│   ├── prophet_model.pkl
│   ├── catboost_model.pkl
│   ├── classifier_model.pkl
│   ├── regime_models.pkl
│   └── quantile_models.pkl
├── outputs/                     # Generated files
│   ├── bitcoin_forecast.png
│   ├── bitcoin_future_predictions.csv
│   ├── backtest_results.png
│   └── bitcoin_forecast.log
└── plots/                       # Visualization outputs
    ├── price_analysis.png
    ├── model_comparison.png
    └── risk_analysis.png
```

## 🚀 Enhanced Features

### 🖥️ Web Dashboard
Launch the interactive web interface:
```bash
python launch_dashboard.py
```
Or manually:
```bash
streamlit run streamlit_dashboard.py
```

Features:
- Interactive parameter configuration
- Real-time model training progress
- Live prediction visualization
- Model performance comparison
- Risk analysis charts

### 📊 Backtesting Framework
Evaluate model performance on historical data:
```python
from backtesting_framework import BitcoinBacktester

backtester = BitcoinBacktester(initial_capital=10000)

# Walk-forward backtesting
results = backtester.walk_forward_backtest(
    forecaster, data, window_size=365, step_size=30
)

# Strategy backtesting
metrics = backtester.strategy_backtest(
    data, predictions, strategy='momentum'
)

# Visualize results
backtester.plot_backtest_results(data)
```

### 📈 Enhanced Visualizations
Advanced charting and analysis:
```python
from visualization_utils import BitcoinVisualizer

viz = BitcoinVisualizer()

# Comprehensive price analysis
viz.plot_price_analysis(data, save_path='analysis.png')

# Interactive prediction charts
viz.plot_predictions_interactive(data, predictions)

# Model comparison
viz.plot_model_comparison(model_results)

# Feature importance analysis
viz.plot_feature_importance(features, importance_values)

# Risk metrics dashboard
viz.plot_risk_metrics(returns, benchmark_returns)
```

### ⚙️ Configuration Management
Use YAML configuration for easy parameter management:
```python
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use configuration in forecaster
btc = UltimateBitcoinForecaster()
btc.fred_api_key = config['api']['fred_key']
```

## 🚀 Advanced Usage

### Hyperparameter Optimization
```python
# Optimize all models automatically
btc.optimize_hyperparameters(X_scaled, y_scaled, dates, lookback=30)
```

### Custom Predictions
```python
# Predict with custom parameters
predictions = btc.predict_with_cycle_awareness(
    lookback=45,  # Look back 45 days
    days_ahead=60  # Predict 60 days ahead
)
```

### Model Persistence
```python
# Models are automatically saved after training
btc.save_models()

# Load models (implement loading functionality)
# btc.load_models()
```

### Batch Processing
```python
# Process multiple prediction horizons
horizons = [7, 14, 30, 60, 90]
for days in horizons:
    preds = btc.predict_with_cycle_awareness(days_ahead=days)
    preds.to_csv(f'predictions_{days}d.csv')
```

## 🐛 Troubleshooting

### Common Issues
1. **API Rate Limits**: Use delays between API calls
2. **Missing Data**: Check API key validity and network connection
3. **GPU Issues**: Ensure CUDA/cuDNN compatibility with TensorFlow
4. **Memory Errors**: Reduce dataset size or use CPU

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data sources
- New ML models
- Better feature engineering
- Performance optimizations
- Web interface
- Real-time monitoring

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- TwelveData for price data API
- Federal Reserve Economic Data (FRED)
- Alternative.me for sentiment data
- Blockchain.com for on-chain metrics
- Open source ML libraries: scikit-learn, TensorFlow, XGBoost, Prophet

---

**⚡ Remember**: Cryptocurrency investing involves substantial risk. Always do your own research and never invest more than you can afford to lose.
