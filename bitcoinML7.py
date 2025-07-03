import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import logging
import os
import pandas_ta as ta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from prophet import Prophet
import optuna
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.python.client import device_lib
import warnings
import pickle
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif




# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Configure logging with StreamHandler for console output
logging.basicConfig(level=logging.INFO, filename="bitcoin_forecast.log", 
                    format='%(asctime)s %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Set random seeds for reproducibility
np.random.seed(42)
random_state = 42

class UltimateBitcoinForecaster:
    def __init__(self):
        """Initialize the Bitcoin forecasting tools with additional metrics storage."""
        self.data = None
        self.xgb_model = None
        self.lstm_model = None
        self.prophet_model = None
        self.cat_model = None
        self.classifier_model = None
        self.scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.stacking_model = None
        self.regime_models = {}
        self.best_xgb_params = None
        self.best_lstm_params = None
        self.best_prophet_params = None
        self.best_cat_params = None
        self.final_rmse = None
        self.final_mape = None
        self.final_dir_acc = None
        self.fred_api_key = None
        self.selected_features = None  # To store selected features for consistency
        self.quantile_models = {}  # To store quantile regression models

    def engineer_directional_features(self):
        """Engineer robust features for directional (up/down) prediction."""
        df = self.data.copy()
        # Price deviation features
        df['Dev_Close_SMA50'] = (df['Close'] - df['SMA50']) / (df['SMA50'] + 1e-8)
        df['Dev_Close_SMA7'] = (df['Close'] - df['SMA7']) / (df['SMA7'] + 1e-8)
        # Ratio of moving averages (momentum shifts)
        df['SMA7_SMA25_ratio'] = df['SMA7'] / (df['SMA25'] + 1e-8)
        # RSI normalized by volatility
        df['RSI_by_ATR'] = df['RSI'] / (df['ATR'] + 1e-8)
        # MACD normalized by ATR
        df['MACD_by_ATR'] = df['MACD'] / (df['ATR'] + 1e-8)
        # Additional features as desired...
        self.data = df

    def safe_feature_selection_direction(self, X, y, n_features=15):
        """Select best features for direction classification."""
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()
    
    def train_directional_classifier(self, X_train, y_direction):
        """Train a classifier to predict next-day direction."""
        from xgboost import XGBClassifier
        self.direction_classifier = XGBClassifier(random_state=42)
        self.direction_classifier.fit(X_train, y_direction)
        logging.info("Directional classifier trained.")

    def check_gpu(self):
        """Ensure GPU is used if available and set memory growth."""
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU detected. Proceeding with CPU.")
            logging.info("No GPU detected. Proceeding with CPU.")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            devices = device_lib.list_local_devices()
            gpu_details = [d for d in devices if d.device_type == 'GPU']
            if gpu_details:
                gpu_name = gpu_details[0].name
                gpu_memory = gpu_details[0].memory_limit / (1024 ** 2)
                print(f"Using GPU: {gpu_name} with {gpu_memory:.0f} MB memory")
                logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.0f} MB memory")
            else:
                print(f"Using GPU: {gpus[0].name}")
                logging.info(f"Using GPU: {gpus[0].name}")

    def fetch_bitcoin_data(self, days=5000, api_key=None):
        """Fetch Bitcoin price history from TwelveData API."""
        if not api_key:
            logging.error("API key is required.")
            print("API key is required.")
            return None
        
        symbol = "BTC/USD"
        interval = "1day"
        max_days = min(days, 5000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': max_days,
            'apikey': api_key
        }
        
        try:
            response = requests.get("https://api.twelvedata.com/time_series", params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'values' not in data:
                logging.error("No 'values' in API response.")
                print("No data found.")
                return None
            
            records = [
                {
                    'Timestamp': pd.to_datetime(item['datetime']),
                    'Open': float(item['open']),
                    'High': float(item['high']),
                    'Low': float(item['low']),
                    'Close': float(item['close']),
                    'Volume': 0
                }
                for item in data['values']
            ]
            
            df = pd.DataFrame(records).set_index('Timestamp').sort_index()
            self.data = df.tail(max_days)
            logging.info(f"Fetched {len(self.data)} days of data.")
            print(f"Fetched {len(self.data)} days of data.")
            return self.data
        
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            print(f"Error fetching data: {e}")
            return None

    def fetch_volume_from_binance(self, days=5000):
        """Fetch volume data from Binance API and merge."""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1d',
            'limit': days
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            volume_df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            volume_df['Timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('Timestamp', inplace=True)
            volume_df = volume_df['volume'].astype(float)
            if self.data is not None:
                self.data['Volume'] = volume_df.reindex(self.data.index, method='nearest')
                logging.info("Volume data fetched and merged.")
                print("Volume data fetched and merged.")
            else:
                logging.error("No existing data to merge volume with.")
                print("No existing data to merge volume with.")
        except Exception as e:
            logging.error(f"Error fetching volume data: {e}")
            print(f"Error fetching volume data: {e}")

    def add_sentiment_data(self, df):
        """Add sentiment data from alternative.me with suffix to avoid overlap."""
        try:
            response = requests.get("https://api.alternative.me/fng?limit=0")
            response.raise_for_status()
            data = response.json()['data']
            sentiment_df = pd.DataFrame(data)
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], unit='s')
            sentiment_df.set_index('timestamp', inplace=True)
            sentiment_df = sentiment_df[['value', 'value_classification']]
            sentiment_df.columns = ['Sentiment_Value', 'Sentiment_Classification']
            df = df.drop(columns=[col for col in sentiment_df.columns if col in df.columns], errors='ignore')
            df = df.join(sentiment_df, how='left', rsuffix='_sentiment')
            df['Sentiment_Value'] = df['Sentiment_Value'].fillna(50)
            df['Sentiment_Classification'] = df['Sentiment_Classification'].fillna('Neutral')
            logging.info("Sentiment data added successfully.")
            return df
        except Exception as e:
            logging.error(f"Error adding sentiment data: {e}")
            print(f"Error adding sentiment data: {e}")
            return df

    def add_macro_data(self, df):
        """Add macroeconomic data from FRED with suffix to avoid overlap."""
        if not self.fred_api_key:
            logging.error("FRED API key is required.")
            print("FRED API key is required.")
            return df
        
        indicators = {
            'DGS10': '10Y_Treasury_Yield',
            'DGS2': '2Y_Treasury_Yield',
            'CPIAUCSL': 'CPI'
        }
        
        for series_id, col_name in indicators.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()['observations']
                macro_df = pd.DataFrame(data)
                macro_df['date'] = pd.to_datetime(macro_df['date'])
                macro_df.set_index('date', inplace=True)
                macro_df = macro_df[['value']].rename(columns={'value': col_name})
                macro_df[col_name] = pd.to_numeric(macro_df[col_name], errors='coerce')
                df = df.drop(columns=[col_name], errors='ignore')
                df = df.join(macro_df, how='left', rsuffix='_macro')
                if col_name == 'CPI':
                    df[col_name] = df[col_name].fillna(method='ffill')
                else:
                    df[col_name] = df[col_name].interpolate()
                logging.info(f"Added {col_name} from FRED.")
            except Exception as e:
                logging.error(f"Error adding {col_name}: {e}")
                print(f"Error adding {col_name}: {e}")
        return df

    def add_onchain_data(self, df):
        """Add on-chain data from Blockchain.com with suffix to avoid overlap."""
        metrics = {
            'n-transactions': 'n_transactions',
            'trade-volume': 'trade_volume_btc',
            'hash-rate': 'hash_rate',
            'difficulty': 'difficulty'
        }
        
        for metric, col_name in metrics.items():
            try:
                url = f"https://api.blockchain.info/charts/{metric}?timespan=all&format=json"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()['values']
                onchain_df = pd.DataFrame(data)
                onchain_df['x'] = pd.to_datetime(onchain_df['x'], unit='s')
                onchain_df.set_index('x', inplace=True)
                onchain_df = onchain_df[['y']].rename(columns={'y': col_name})
                df = df.drop(columns=[col_name], errors='ignore')
                df = df.join(onchain_df, how='left', rsuffix='_onchain')
                df[col_name] = df[col_name].fillna(method='ffill')
                logging.info(f"Added {col_name} from Blockchain.com.")
            except Exception as e:
                logging.error(f"Error adding {col_name}: {e}")
                print(f"Error adding {col_name}: {e}")
        return df

    def add_cycle_features(self):
        """Add Bitcoin halving cycle features."""
        df = self.data.copy()
        halving_dates = [
            pd.Timestamp('2012-11-28'),
            pd.Timestamp('2016-07-09'),
            pd.Timestamp('2020-05-11'),
            pd.Timestamp('2024-04-19')
        ]
        
        df['HalvingCycle'] = 0
        df['DaysSinceHalving'] = 0
        df['DaysToNextHalving'] = 0
        
        for i, date in enumerate(halving_dates):
            if i < len(halving_dates) - 1:
                next_date = halving_dates[i + 1]
                mask = (df.index >= date) & (df.index < next_date)
                df.loc[mask, 'HalvingCycle'] = i + 1
                df.loc[mask, 'DaysSinceHalving'] = (df.index[mask] - date).days
                df.loc[mask, 'DaysToNextHalving'] = (next_date - df.index[mask]).days
            else:
                mask = df.index >= date
                df.loc[mask, 'HalvingCycle'] = i + 1
                df.loc[mask, 'DaysSinceHalving'] = (df.index[mask] - date).days
                next_approx = date + pd.Timedelta(days=1460)
                df.loc[mask, 'DaysToNextHalving'] = (next_approx - df.index[mask]).days
        
        first_halving = halving_dates[0]
        mask_before = df.index < first_halving
        df.loc[mask_before, 'HalvingCycle'] = 0
        df.loc[mask_before, 'DaysSinceHalving'] = (df.index[mask_before] - pd.Timestamp('2009-01-03')).days
        df.loc[mask_before, 'DaysToNextHalving'] = (first_halving - df.index[mask_before]).days
        
        avg_cycle_length = 1460
        df['CyclePosition'] = df['DaysSinceHalving'] / avg_cycle_length
        df['CyclePosition_sin'] = np.sin(2 * np.pi * df['CyclePosition'])
        df['CyclePosition_cos'] = np.cos(2 * np.pi * df['CyclePosition'])
        
        self.data = df
        logging.info("Added halving cycle features to dataset")
        return df

    def calculate_indicators(self):
        """Calculate technical indicators and add external data with robust NaN and inf handling."""
        if self.data is None:
            logging.error("No data available for indicators.")
            print("No data available. Please run fetch_bitcoin_data() first.")
            return None
        
        df = self.data.copy()
        print("Calculating technical indicators...")
        
        if 'Volume' not in df.columns or (df['Volume'] == 0).all():
            print("Volume data is missing or all zeros. Skipping volume-based indicators.")
            volume_based_indicators = ['OBV', 'CMF', 'VWAP', 'ForceIndex13']
        else:
            volume_based_indicators = []
        
        df['LogClose'] = np.log(df['Close'] + 1e-10)  # Prevent log(0) or negative
        df['LogReturns'] = df['LogClose'].diff()
        
        df['SMA7'] = ta.sma(df['Close'], length=7)
        df['SMA25'] = ta.sma(df['Close'], length=25)
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['EMA12'] = ta.ema(df['Close'], length=12)
        df['EMA26'] = ta.ema(df['Close'], length=26)
        
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ROC'] = ta.roc(df['Close'], length=14)
        
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        df['StochK'] = stoch['STOCHk_14_3_3']
        df['StochD'] = stoch['STOCHd_14_3_3']
        
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'] = bbands['BBU_20_2.0']
        df['BB_Middle'] = bbands['BBM_20_2.0']
        df['BB_Lower'] = bbands['BBL_20_2.0']
        
        if 'OBV' not in volume_based_indicators:
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
        if 'CMF' not in volume_based_indicators:
            df['CMF'] = ta.adosc(df['High'], df['Low'], df['Close'], df['Volume'], fast=3, slow=10)
        if 'VWAP' not in volume_based_indicators:
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        if 'ForceIndex13' not in volume_based_indicators:
            df['ForceIndex'] = df['Close'].diff(1) * df['Volume']
            df['ForceIndex13'] = ta.ema(df['ForceIndex'], length=13)
        
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
        
        dmi = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        df['ADX'] = dmi['ADX_14']
        df['DI+'] = dmi['DMP_14']
        df['DI-'] = dmi['DMN_14']
        
        keltner = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
        df['KC_Upper'] = keltner['KCUe_20_2.0']
        df['KC_Lower'] = keltner['KCLe_20_2.0']
        
        df['Returns'] = df['Close'].pct_change()
        df['BullishRegime'] = (df['Close'] > df['SMA50']).astype(int)
        df['Volatility'] = df['Close'].rolling(window=14).std()
        df['DC_Upper'] = df['High'].rolling(window=20).max()
        df['DC_Lower'] = df['Low'].rolling(window=20).min()
        
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        df['BullEngulf'] = 0
        df.loc[(df['Close'].shift(1) < df['Open'].shift(1)) & 
               (df['Close'] > df['Open']) & 
               (df['Close'] > df['Open'].shift(1)) & 
               (df['Open'] < df['Close'].shift(1)), 'BullEngulf'] = 100
        df.loc[(df['Close'].shift(1) > df['Open'].shift(1)) & 
               (df['Close'] < df['Open']) & 
               (df['Close'] < df['Open'].shift(1)) & 
               (df['Open'] > df['Close'].shift(1)), 'BullEngulf'] = -100
        
        # Add external data with safeguards against column overlap
        df = self.add_sentiment_data(df)
        df = self.add_macro_data(df)
        df = self.add_onchain_data(df)
        
        # Drop any duplicate or unnecessary columns
        df = df.loc[:, ~df.columns.str.contains(r'_onchain|_macro|_sentiment\.\d+$', regex=True)]
        
        # Robust NaN and infinite value handling
        final_nan_check = df.isnull().sum()
        if final_nan_check.sum() > 0:
            logging.warning(f"Final NaN check found missing values:\n{final_nan_check[final_nan_check > 0]}")
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
            df.dropna(inplace=True)  # Final cleanup
        
        # Check and handle infinite values
        infinite_check = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if infinite_check.sum() > 0:
            logging.warning(f"Infinite values found in columns:\n{infinite_check[infinite_check > 0]}")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
        
        self.data = df
        self.analyze_feature_correlations(df)
        logging.info(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        print(f"Added {len(df.columns)} indicators. Data shape: {df.shape}")
        return df

    def analyze_feature_correlations(self, df):
        """Analyze feature correlations and check for constant features."""
        numerical_df = df.select_dtypes(include=[np.number])
        target_corr = numerical_df.corr()['Close'].sort_values(ascending=False)
        logging.info(f"Top 10 correlated features: {target_corr[:10].to_dict()}")
        
        corr_matrix = numerical_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(upper.index[i], upper.columns[j], upper.iloc[i, j]) 
                     for i, j in zip(*np.where(upper > 0.95))]
        if high_corr:
            logging.info(f"Highly correlated features (>0.95): {high_corr}")
        
        constant_features = [col for col in numerical_df.columns if numerical_df[col].nunique() == 1]
        if constant_features:
            logging.warning(f"Constant features found: {constant_features}")
            print(f"Constant features found: {constant_features}")

    def detect_market_regimes(self, window=90):
        """Detect market regimes."""
        df = self.data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=window).std()
        
        if 'ADX' not in df.columns:
            adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            df['ADX'] = adx['ADX_14']
        
        df['Regime'] = 0
        bull_condition = (df['Returns'].rolling(window=window).mean() > 0) & \
                         (df['Volatility'] > df['Volatility'].rolling(window=window*3).mean()) & \
                         (df['ADX'] > 25)
        df.loc[bull_condition, 'Regime'] = 1
        bear_condition = (df['Returns'].rolling(window=window).mean() < 0) & \
                         (df['Volatility'] > df['Volatility'].rolling(window=window*3).mean()) & \
                         (df['ADX'] > 25)
        df.loc[bear_condition, 'Regime'] = 2
        accum_condition = (df['Volatility'] < df['Volatility'].rolling(window=window*3).mean()) & \
                          (df['ADX'] < 20)
        df.loc[accum_condition, 'Regime'] = 3
        
        df['Regime'] = df['Regime'].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        self.data = df
        logging.info("Detected market regimes based on price action")
        return df

    def safe_feature_selection(self, X, y, n_features=30):
        """Perform feature selection with error handling."""
        try:
            constant_features = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_features:
                logging.warning(f"Dropping constant features: {constant_features}")
                X = X.drop(columns=constant_features)
            
            if len(X.columns) <= n_features:
                return X.columns.tolist()
            
            model = xgb.XGBRegressor(random_state=random_state)
            tscv = TimeSeriesSplit(n_splits=3)
            rfecv = RFECV(estimator=model, step=0.1, cv=tscv, scoring='neg_mean_squared_error')
            rfecv.fit(X, y)
            selected_features = X.columns[rfecv.support_].tolist()
            
            if len(selected_features) > n_features:
                feature_ranking = pd.Series(rfecv.ranking_, index=X.columns)
                selected_features = feature_ranking.nsmallest(n_features).index.tolist()
            elif len(selected_features) == 0:
                logging.warning("RFECV selected no features. Using all features instead.")
                selected_features = X.columns.tolist()
            
            logging.info(f"Selected {len(selected_features)} features with RFECV.")
            return selected_features
            
        except Exception as e:
            logging.error(f"Feature selection failed: {e}")
            target_corr = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = target_corr.head(n_features).index.tolist()
            return selected_features

    def prepare_ml_data(self, lookback=30):
        """Prepare features and target with lagged features, preventing look-ahead bias."""
        if self.data is None:
            logging.error("No indicators available.")
            print("No indicators available. Run calculate_indicators() first.")
            return None

        df = self.data.copy()
        print("Preparing ML data with lookback...")

        # Create lagged features first to prevent look-ahead bias
        for lag in range(1, lookback + 1):
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)

        # Drop rows with NaNs in lagged features
        df = df.dropna(subset=[f'Close_lag_{lag}' for lag in range(1, lookback + 1)] + 
                       [f'Volume_lag_{lag}' for lag in range(1, lookback + 1)])
        
        if df.empty:
            logging.error("No data left after dropping NaNs in lagged features.")
            print("Error: No data left after dropping NaNs in lagged features.")
            return None

        # Compute target after lags to avoid leakage
        df['Target'] = df['Close'].pct_change().shift(-1)

        # Drop rows where target is NaN
        df = df.dropna(subset=['Target'])
        
        if df.empty:
            logging.error("No data left after dropping NaNs in target.")
            print("Error: No data left after dropping NaNs in target.")
            return None

        # Define features
        base_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'SMA7', 'SMA25', 'SMA50', 'EMA12', 'EMA26',
            'MACD', 'MACD_Signal', 'MACD_Hist', 'RSI', 'ROC', 'StochK', 'StochD', 'ATR',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CMF', 'VWAP', 'ForceIndex13', 'CCI',
            'ADX', 'DI+', 'DI-', 'KC_Upper', 'KC_Lower', 'Returns', 'LogReturns', 'BullishRegime',
            'Volatility', 'DC_Upper', 'DC_Lower', 'BullEngulf', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'Month_sin', 'Month_cos', 'LogClose',
            'HalvingCycle', 'DaysSinceHalving', 'DaysToNextHalving', 'CyclePosition_sin', 'CyclePosition_cos',
            'Sentiment_Value', '10Y_Treasury_Yield', '2Y_Treasury_Yield', 'CPI',
            'n_transactions', 'trade_volume_btc', 'hash_rate', 'difficulty'
        ]
        available_features = [f for f in base_features if f in df.columns]
        features = available_features + [f'Close_lag_{lag}' for lag in range(1, lookback + 1)] + \
                   [f'Volume_lag_{lag}' for lag in range(1, lookback + 1)]
        
        X = df[features]
        y = df['Target']
        
        # Perform feature selection
        self.selected_features = self.safe_feature_selection(X, y, n_features=30)
        X_selected = X[self.selected_features]
        
        # Fit scaler on the selected features once
        X_scaled_selected = self.scaler.fit_transform(X_selected)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        logging.info(f"Prepared {len(X_scaled_selected)} samples with {len(self.selected_features)} features.")
        print(f"Prepared {len(X_scaled_selected)} samples with {len(self.selected_features)} features.")
        return X_scaled_selected, y_scaled, df.index, self.selected_features

    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost hyperparameters."""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': random_state
            }
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_val_inv = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_val_inv, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200)
        self.best_xgb_params = study.best_params
        return study.best_params

    def optimize_lstm(self, X_train, y_train, X_val, y_val, lookback):
        """Optimize LSTM hyperparameters."""
        def objective(trial):
            units1 = trial.suggest_int('units1', 64, 256)
            units2 = trial.suggest_int('units2', 32, 128)
            units3 = trial.suggest_int('units3', 16, 64)
            dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
            dropout2 = trial.suggest_float('dropout2', 0.1, 0.5)
            
            X_lstm_train, y_lstm_train = self.prepare_lstm_data(X_train, y_train, lookback)
            X_lstm_val, y_lstm_val = self.prepare_lstm_data(X_val, y_val, lookback)
            
            inputs = Input(shape=(lookback, X_train.shape[1]))
            lstm1 = Bidirectional(LSTM(units1, return_sequences=True))(inputs)
            dropout_layer1 = Dropout(dropout1)(lstm1)
            lstm2 = Bidirectional(LSTM(units2, return_sequences=True))(dropout_layer1)
            attention_output = Attention()([lstm2, lstm2])
            dropout_layer2 = Dropout(dropout2)(attention_output)
            lstm3 = LSTM(units3)(dropout_layer2)
            outputs = Dense(1, activation='linear')(lstm3)
            
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='nadam', loss='mse')
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=64, 
                      validation_data=(X_lstm_val, y_lstm_val), callbacks=[early_stopping], verbose=0)
            
            preds_scaled = model.predict(X_lstm_val).flatten()
            preds = self.target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            y_val_inv = self.target_scaler.inverse_transform(y_lstm_val.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_val_inv, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        self.best_lstm_params = study.best_params
        return study.best_params

    def optimize_prophet(self, df_train, df_val):
        """Optimize Prophet hyperparameters."""
        def objective(trial):
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
            }
            model = Prophet(**params)
            model.add_seasonality(name='halving_cycle', period=1460, fourier_order=5)
            model.fit(df_train)
            forecast = model.predict(df_val)
            mse = mean_squared_error(df_val['y'], forecast['yhat'])
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        self.best_prophet_params = study.best_params
        return study.best_params

    def optimize_catboost(self, X_train, y_train, X_val, y_val):
        """Optimize CatBoost hyperparameters."""
        def objective(trial):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': random_state,
                'verbose': 0
            }
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_val_inv = self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            mse = mean_squared_error(y_val_inv, preds)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200)
        self.best_cat_params = study.best_params
        return study.best_params

    def prepare_lstm_data(self, X, y, lookback):
        """Prepare data for LSTM with lookback window."""
        X_lstm = []
        y_lstm = []
        for i in range(lookback, len(X)):
            X_lstm.append(X[i-lookback:i])
            y_lstm.append(y[i])
        return np.array(X_lstm), np.array(y_lstm)

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with selected features."""
        if self.best_xgb_params is None:
            self.best_xgb_params = {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 100, 'subsample': 0.8, 'random_state': random_state}
        self.xgb_model = xgb.XGBRegressor(**self.best_xgb_params)
        self.xgb_model.fit(X_train, y_train)
        logging.info("XGBoost model trained.")

    def train_lstm(self, X_train, y_train, lookback):
        """Train LSTM model with selected features."""
        if self.best_lstm_params is None:
            self.best_lstm_params = {'units1': 128, 'units2': 64, 'units3': 32, 'dropout1': 0.2, 'dropout2': 0.2}
        
        X_lstm, y_lstm = self.prepare_lstm_data(X_train, y_train, lookback)
        inputs = Input(shape=(lookback, X_train.shape[1]))
        lstm1 = Bidirectional(LSTM(self.best_lstm_params['units1'], return_sequences=True))(inputs)
        dropout1 = Dropout(self.best_lstm_params['dropout1'])(lstm1)
        lstm2 = Bidirectional(LSTM(self.best_lstm_params['units2'], return_sequences=True))(dropout1)
        attention_output = Attention()([lstm2, lstm2])
        dropout2 = Dropout(self.best_lstm_params['dropout2'])(attention_output)
        lstm3 = LSTM(self.best_lstm_params['units3'])(dropout2)
        outputs = Dense(1, activation='linear')(lstm3)
        
        self.lstm_model = Model(inputs=inputs, outputs=outputs)
        self.lstm_model.compile(optimizer='nadam', loss='mse')
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=64, callbacks=[early_stopping], verbose=0)
        logging.info("LSTM model trained.")

    def train_prophet(self, df_train):
        """Train Prophet model with best parameters."""
        if self.best_prophet_params is None:
            self.best_prophet_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            }
        self.prophet_model = Prophet(**self.best_prophet_params)
        self.prophet_model.add_seasonality(name='halving_cycle', period=1460, fourier_order=5)
        self.prophet_model.fit(df_train)
        logging.info("Prophet model trained.")

    def train_catboost(self, X_train, y_train):
        """Train CatBoost model with selected features."""
        if self.best_cat_params is None:
            self.best_cat_params = {'depth': 6, 'learning_rate': 0.1, 'iterations': 500, 'l2_leaf_reg': 3, 'random_seed': random_state}
        self.cat_model = CatBoostRegressor(**self.best_cat_params, verbose=0)
        self.cat_model.fit(X_train, y_train)
        logging.info("CatBoost model trained.")

    def train_classifier(self, X_train, y_train):
        """Train a classifier for direction prediction."""
        y_direction = np.where(y_train > 0, 1, 0)
        self.classifier_model = xgb.XGBClassifier(random_state=random_state)
        self.classifier_model.fit(X_train, y_direction)
        logging.info("Classifier model trained.")

    def train_regime_specific_models(self, X_train, y_train, lookback):
        """Train regime-specific XGBoost models using consistent selected features."""
        df = self.data.iloc[-len(X_train):].copy()
        regimes = df['Regime'].unique()
        regime_models = {}
        
        for regime in regimes:
            regime_idx = df[df['Regime'] == regime].index
            regime_train_idx = [i for i, idx in enumerate(df.index) if idx in regime_idx]
            if len(regime_train_idx) < lookback + 1:
                continue
            X_regime = X_train[regime_train_idx]
            y_regime = y_train[regime_train_idx]
            model = xgb.XGBRegressor(**self.best_xgb_params)
            model.fit(X_regime, y_regime)
            regime_models[regime] = model
            logging.info(f"Trained regime-specific model for regime {regime}")
        return regime_models

    def optimize_hyperparameters(self, X_scaled, y_scaled, dates, lookback):
        """Optimize hyperparameters for all models using the same selected features."""
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        dates_train = dates[:split_idx]
        dates_val = dates[split_idx:]
        
        print("Optimizing XGBoost...")
        self.optimize_xgboost(X_train, y_train, X_val, y_val)
        print("Optimizing LSTM...")
        self.optimize_lstm(X_train, y_train, X_val, y_val, lookback)
        print("Optimizing Prophet...")
        df_train = pd.DataFrame({'ds': dates_train, 'y': self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()})
        df_val = pd.DataFrame({'ds': dates_val, 'y': self.target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()})
        self.optimize_prophet(df_train, df_val)
        print("Optimizing CatBoost...")
        self.optimize_catboost(X_train, y_train, X_val, y_val)

    def get_aligned_predictions(self, X_test, dates_test, lookback):
        """Get predictions from all models aligned by date, using selected features."""
        X_lstm, _ = self.prepare_lstm_data(X_test, np.zeros(len(X_test)), lookback)
        
        xgb_pred = self.xgb_model.predict(X_test)
        lstm_pred = self.lstm_model.predict(X_lstm).flatten()
        prophet_df = pd.DataFrame({'ds': dates_test})
        prophet_forecast = self.prophet_model.predict(prophet_df)
        prophet_pred = prophet_forecast['yhat'].values
        cat_pred = self.cat_model.predict(X_test)
        classifier_pred = self.classifier_model.predict(X_test)
        
        min_len = min(len(xgb_pred[lookback:]), len(lstm_pred), len(prophet_pred[lookback:]), len(cat_pred[lookback:]), len(classifier_pred[lookback:]))
        
        predictions = {
            'XGBoost': xgb_pred[lookback:lookback+min_len],
            'LSTM': lstm_pred[:min_len],
            'Prophet': prophet_pred[lookback:lookback+min_len],
            'CatBoost': cat_pred[lookback:lookback+min_len],
            'Classifier': classifier_pred[lookback:lookback+min_len],
            'dates': dates_test[lookback:lookback+min_len]
        }
        return predictions

    def train_all_models(self, test_size=0.1, lookback=30):
        """Train all models with improved stacking approach and NaN debugging."""
        X_scaled, y_scaled, dates, selected_features = self.prepare_ml_data(lookback)
        if X_scaled is None:
            return None
        
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = {'XGBoost': [], 'LSTM': [], 'Prophet': [], 'CatBoost': [], 'Ensemble': []}
        
        for train_idx, test_idx in tscv.split(X_scaled):
            if len(test_idx) < lookback:
                continue
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
            dates_train, dates_test = dates[train_idx], dates[test_idx]
            
            self.train_xgboost(X_train, y_train)
            self.train_lstm(X_train, y_train, lookback)
            df_train = pd.DataFrame({'ds': dates_train, 'y': self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()})
            self.train_prophet(df_train)
            self.train_catboost(X_train, y_train)
            self.train_classifier(X_train, y_train)
            
            predictions = self.get_aligned_predictions(X_test, dates_test, lookback)
            
            if all(model in predictions for model in ['XGBoost', 'LSTM', 'Prophet', 'CatBoost']):
                weights = {'XGBoost': 0.3, 'LSTM': 0.25, 'Prophet': 0.25, 'CatBoost': 0.2}
                ensemble_pred = sum(weights[model] * predictions[model] for model in weights.keys())
                
                y_test_aligned = y_test[lookback:lookback+len(ensemble_pred)]
                if len(y_test_aligned) > 0:
                    y_test_inv = self.target_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
                    ensemble_pred_inv = self.target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
                    
                    # Debugging NaN values
                    print("--- NaN Investigation (CV Fold) ---")
                    print(f"Shape of y_test_aligned: {y_test_aligned.shape}")
                    print(f"Shape of ensemble_pred: {ensemble_pred.shape}")
                    print(f"Any NaNs in y_test_inv? {np.isnan(y_test_inv).any()}")
                    print(f"Any NaNs in ensemble_pred_inv? {np.isnan(ensemble_pred_inv).any()}")
                    
                    if np.isnan(ensemble_pred_inv).any():
                        print("NaNs found in ensemble predictions. Locating source...")
                        xgb_pred_inv = self.target_scaler.inverse_transform(predictions['XGBoost'].reshape(-1, 1))
                        lstm_pred_inv = self.target_scaler.inverse_transform(predictions['LSTM'].reshape(-1, 1))
                        prophet_pred_inv = self.target_scaler.inverse_transform(predictions['Prophet'].reshape(-1, 1))
                        cat_pred_inv = self.target_scaler.inverse_transform(predictions['CatBoost'].reshape(-1, 1))
                        print(f"  NaNs in XGBoost: {np.isnan(xgb_pred_inv).any()}")
                        print(f"  NaNs in LSTM: {np.isnan(lstm_pred_inv).any()}")
                        print(f"  NaNs in Prophet: {np.isnan(prophet_pred_inv).any()}")
                        print(f"  NaNs in CatBoost: {np.isnan(cat_pred_inv).any()}")
                    
                    if not np.isnan(y_test_inv).any() and not np.isnan(ensemble_pred_inv).any():
                        rmse = np.sqrt(mean_squared_error(y_test_inv, ensemble_pred_inv))
                        mape = mean_absolute_percentage_error(y_test_inv + 1e-10, ensemble_pred_inv + 1e-10)
                        dir_acc = np.mean(np.sign(ensemble_pred_inv) == np.sign(y_test_inv))
                        cv_scores['Ensemble'].append({'RMSE': rmse, 'MAPE': mape, 'Dir_Acc': dir_acc})
                    else:
                        logging.error("NaNs detected in CV fold. Skipping metrics.")
        
        for model, scores in cv_scores.items():
            if scores:
                avg_rmse = np.mean([s['RMSE'] for s in scores])
                avg_mape = np.mean([s['MAPE'] for s in scores])
                avg_dir_acc = np.mean([s['Dir_Acc'] for s in scores])
                print(f"{model} - Avg RMSE: {avg_rmse:.4f}, Avg MAPE: {avg_mape:.4f}, Avg Dir Acc: {avg_dir_acc:.4f}")
        
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        self.train_xgboost(X_train, y_train)
        self.train_lstm(X_train, y_train, lookback)
        df_train = pd.DataFrame({'ds': dates_train, 'y': self.target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()})
        self.train_prophet(df_train)
        self.train_catboost(X_train, y_train)
        self.train_classifier(X_train, y_train)
        self.regime_models = self.train_regime_specific_models(X_train, y_train, lookback)
        
        # Train quantile models with tuned parameters
        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            model = LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=400,
                min_child_samples=40,
                learning_rate=0.05,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            self.quantile_models[q] = model
        logging.info(f"Trained quantile models for {quantiles}")
        
        predictions = self.get_aligned_predictions(X_test, dates_test, lookback)
        
        weights = {'XGBoost': 0.3, 'LSTM': 0.25, 'Prophet': 0.25, 'CatBoost': 0.2}
        ensemble_pred = sum(weights[model] * predictions[model] for model in weights.keys())
        
        y_test_aligned = y_test[lookback:lookback+len(ensemble_pred)]
        y_test_inv = self.target_scaler.inverse_transform(y_test_aligned.reshape(-1, 1)).flatten()
        ensemble_pred_inv = self.target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
        
        # Debugging NaN values in final evaluation
        print("--- Final NaN Investigation ---")
        print(f"Shape of y_test_inv: {y_test_inv.shape}")
        print(f"Shape of ensemble_pred_inv: {ensemble_pred_inv.shape}")
        print(f"Any NaNs in y_test_inv? {np.isnan(y_test_inv).any()}")
        print(f"Any NaNs in ensemble_pred_inv? {np.isnan(ensemble_pred_inv).any()}")
        
        if np.isnan(ensemble_pred_inv).any():
            print("NaNs found in final ensemble predictions. Locating source...")
            xgb_pred_inv = self.target_scaler.inverse_transform(predictions['XGBoost'].reshape(-1, 1))
            lstm_pred_inv = self.target_scaler.inverse_transform(predictions['LSTM'].reshape(-1, 1))
            prophet_pred_inv = self.target_scaler.inverse_transform(predictions['Prophet'].reshape(-1, 1))
            cat_pred_inv = self.target_scaler.inverse_transform(predictions['CatBoost'].reshape(-1, 1))
            print(f"  NaNs in XGBoost: {np.isnan(xgb_pred_inv).any()}")
            print(f"  NaNs in LSTM: {np.isnan(lstm_pred_inv).any()}")
            print(f"  NaNs in Prophet: {np.isnan(prophet_pred_inv).any()}")
            print(f"  NaNs in CatBoost: {np.isnan(cat_pred_inv).any()}")
        
        if not np.isnan(y_test_inv).any() and not np.isnan(ensemble_pred_inv).any():
            self.final_rmse = np.sqrt(mean_squared_error(y_test_inv, ensemble_pred_inv))
            self.final_mape = mean_absolute_percentage_error(y_test_inv + 1e-10, ensemble_pred_inv + 1e-10)
            self.final_dir_acc = np.mean(np.sign(ensemble_pred_inv) == np.sign(y_test_inv))
            
            print("\n--- Final Model Performance on Test Set ---")
            print(f"RMSE: {self.final_rmse:.4f}")
            print(f"MAPE: {self.final_mape:.4f}")
            print(f"Directional Accuracy: {self.final_dir_acc:.4f}")
            
            logging.info(f"Final Ensemble RMSE: {self.final_rmse:.4f}, MAPE: {self.final_mape:.4f}, Directional Accuracy: {self.final_dir_acc:.4f}")
            
            self.plot_predictions(dates_test[lookback:lookback+len(ensemble_pred)], y_test_inv, ensemble_pred_inv)
            self.save_models()
        else:
            logging.error("NaNs found in final predictions or ground truth. Skipping metrics and saving.")
        
        return X_test[lookback:], y_test_inv, ensemble_pred_inv, dates_test[lookback:]

    def plot_predictions(self, dates, actual, predicted):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(14, 7))
        plt.plot(dates, actual, label='Actual Returns', color='blue')
        plt.plot(dates, predicted, label='Predicted Returns', color='orange')
        plt.title('Bitcoin Returns: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('bitcoin_forecast.png')
        plt.close()
        logging.info("Prediction plot saved as bitcoin_forecast.png")

    def save_models(self):
        """Save trained models to disk."""
        with open('xgb_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        self.lstm_model.save('lstm_model.h5')
        with open('prophet_model.pkl', 'wb') as f:
            pickle.dump(self.prophet_model, f)
        with open('cat_model.pkl', 'wb') as f:
            pickle.dump(self.cat_model, f)
        with open('classifier_model.pkl', 'wb') as f:
            pickle.dump(self.classifier_model, f)
        with open('regime_models.pkl', 'wb') as f:
            pickle.dump(self.regime_models, f)
        with open('quantile_models.pkl', 'wb') as f:
            pickle.dump(self.quantile_models, f)
        logging.info("All models saved to disk.")

    def calculate_basic_indicators(self):
        """Calculate only essential indicators for prediction."""
        df = self.data
        
        if len(df) > 50:
            recent_df = df.tail(50).copy()
            
            recent_df['SMA7'] = recent_df['Close'].rolling(7, min_periods=1).mean()
            recent_df['SMA25'] = recent_df['Close'].rolling(25, min_periods=1).mean()
            recent_df['RSI'] = ta.rsi(recent_df['Close'], length=14)
            
            self.data.loc[recent_df.index, ['SMA7', 'SMA25', 'RSI']] = recent_df[['SMA7', 'SMA25', 'RSI']]

    def predict_with_cycle_awareness(self, lookback=30, days_ahead=30):
        """Generate future predictions with proper feature consistency and quantile-based confidence intervals."""
        if self.data is None or self.xgb_model is None:
            print("Models or data not available. Train models first.")
            return None
        
        original_data_len = len(self.data)
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')
        
        predictions = []
        lower_preds = []
        upper_preds = []
        
        for i in range(days_ahead):
            for lag in range(1, lookback + 1):
                lag_col = f'Close_lag_{lag}'
                if lag_col in self.selected_features:
                    self.data[lag_col] = self.data['Close'].shift(lag)

            recent_data = self.data.tail(lookback)
            
            if hasattr(self, 'selected_features') and self.selected_features:
                missing_features = set(self.selected_features) - set(recent_data.columns)
                if missing_features:
                    print(f"Warning: Missing features {missing_features}. Using available features.")
                    available_features = [f for f in self.selected_features if f in recent_data.columns]
                    if not available_features:
                        print("Error: No valid features available for prediction.")
                        break
                    features_to_use = available_features
                else:
                    features_to_use = self.selected_features
            else:
                basic_features = ['Close', 'Volume', 'SMA7', 'SMA25', 'RSI', 'MACD']
                features_to_use = [f for f in basic_features if f in recent_data.columns]
            
            last_X = recent_data[features_to_use].iloc[-1:].values
            
            try:
                last_X_scaled = self.scaler.transform(last_X)
            except ValueError as e:
                print(f"Scaling error: {e}")
                break
            
            current_regime = self.data['Regime'].iloc[-1] if 'Regime' in self.data.columns else 0
            model = self.regime_models.get(current_regime, self.xgb_model) if self.regime_models else self.xgb_model
            
            pred_scaled = model.predict(last_X_scaled)[0]
            pred = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred)
            
            lower_pred_scaled = self.quantile_models[0.1].predict(last_X_scaled)[0]
            upper_pred_scaled = self.quantile_models[0.9].predict(last_X_scaled)[0]
            lower_pred = self.target_scaler.inverse_transform([[lower_pred_scaled]])[0][0]
            upper_pred = self.target_scaler.inverse_transform([[upper_pred_scaled]])[0][0]
            lower_preds.append(lower_pred)
            upper_preds.append(upper_pred)
            
            new_close = recent_data['Close'].iloc[-1] * (1 + pred)
            
            new_row = pd.Series(index=self.data.columns, dtype=float)
            new_row['Close'] = new_close
            new_row['Open'] = new_close
            new_row['High'] = new_close * 1.02
            new_row['Low'] = new_close * 0.98
            new_row['Volume'] = recent_data['Volume'].iloc[-1]
            new_row.name = future_dates[i]
            
            self.data = pd.concat([self.data, pd.DataFrame([new_row])])
            self.calculate_basic_indicators()
        
        self.data = self.data.iloc[:original_data_len]
        
        future_df = pd.DataFrame({
            'Date': future_dates[:len(predictions)],
            'Predicted_Returns': predictions,
            'Lower_Return': lower_preds,
            'Upper_Return': upper_preds
        })
        
        last_known_price = self.data['Close'].iloc[-1]
        future_df['Predicted_Price'] = last_known_price * (1 + pd.Series(predictions)).cumprod()
        future_df['Lower_Price'] = last_known_price * (1 + future_df['Lower_Return']).cumprod()
        future_df['Upper_Price'] = last_known_price * (1 + future_df['Upper_Return']).cumprod()
        
        print("\n--- Model Performance on Test Set ---")
        print(f"RMSE: {self.final_rmse:.4f}")
        print(f"MAPE: {self.final_mape:.4f}")
        print(f"Directional Accuracy: {self.final_dir_acc:.4f}")
        print("\n**Disclaimer**: Predicting cryptocurrency prices is extremely challenging due to their high volatility and unpredictable factors. Use with caution.")
        
        print("\n--- Future Predictions ---")
        print(future_df[['Date', 'Predicted_Returns', 'Predicted_Price', 'Lower_Price', 'Upper_Price']])
        
        future_df.to_csv('bitcoin_future_predictions.csv', index=False)
        print("\nPredictions saved to 'bitcoin_future_predictions.csv'")
        
        return future_df

    def validate_prediction_stability(self, n_runs=5, days_ahead=5):
        """Test prediction consistency across multiple runs with slight perturbations."""
        predictions_list = []
        original_data = self.data.copy()
        
        for run in range(n_runs):
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            noise = np.random.normal(0, 0.001, size=(len(self.data), len(numeric_cols)))
            self.data[numeric_cols] += noise
            
            pred_df = self.predict_with_cycle_awareness(lookback=30, days_ahead=days_ahead)
            predictions_list.append(pred_df['Predicted_Returns'].values)
            
            self.data = original_data.copy()
        
        pred_array = np.array(predictions_list)
        stability_score = 1 - np.std(pred_array, axis=0).mean()
        
        print(f"Prediction Stability Score: {stability_score:.4f}")
        return stability_score

    def detect_regime_changes(self, window=20):
        """Detect potential regime changes based on volatility ratios."""
        recent_returns = self.data['Returns'].tail(window)
        recent_volatility = recent_returns.std()
        long_term_volatility = self.data['Returns'].tail(window*3).std()
        
        volatility_ratio = recent_volatility / long_term_volatility
        
        if volatility_ratio > 1.5:
            return "High Volatility Regime"
        elif volatility_ratio < 0.5:
            return "Low Volatility Regime"
        else:
            return "Stable Regime"

if __name__ == "__main__":
    btc = UltimateBitcoinForecaster()
    btc.check_gpu()
    btc.fred_api_key = 'b94daa8d4633928f2faa1b669f07eef3'
    btc.fetch_bitcoin_data(days=5000, api_key='f0872d0dd0a841788d2694eb4ec0b4e0')
    btc.fetch_volume_from_binance(days=5000)
    if btc.data is not None:
        btc.add_cycle_features()
        btc.calculate_indicators()
        btc.engineer_directional_features()
        btc.detect_market_regimes()

        # --- BEGIN DIRECTIONAL PIPELINE PATCH ---
        lookback = 30
        ml_out = btc.prepare_ml_data(lookback=lookback)
        if ml_out is None:
            exit(1)
        X_all, y_all, dates, _ = ml_out
        
        # Create binary direction target: 1 if up, 0 if down or flat
        y_direction = (y_all > 0).astype(int)
        X_df = pd.DataFrame(X_all, columns=btc.selected_features)

        # Feature selection for classification
        selected_directional_feats = btc.safe_feature_selection_direction(X_df, y_direction, n_features=15)
        X_selected = X_df[selected_directional_feats]

        # TimeSeriesSplit for robust validation
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        tscv = TimeSeriesSplit(n_splits=3)
        acc_scores = []
        print("\nTime Series CV directional accuracy breakdown:")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected), 1):
            btc.train_directional_classifier(X_selected.iloc[train_idx], y_direction[train_idx])
            y_pred = btc.direction_classifier.predict(X_selected.iloc[test_idx])
            acc = accuracy_score(y_direction[test_idx], y_pred)
            acc_scores.append(acc)
            print(f"  Fold {fold}: accuracy = {acc:.4f}")

        print(f"\nMean CV Directional Accuracy: {np.mean(acc_scores):.4f}")

        # Final train/test evaluation
        split_idx = int(0.8 * len(X_selected))
        X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
        y_train, y_test = y_direction[:split_idx], y_direction[split_idx:]

        btc.train_directional_classifier(X_train, y_train)
        y_pred = btc.direction_classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nDirectional Accuracy (hold-out test set): {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=3))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        # (Optional) Print most important features
        importances = btc.direction_classifier.feature_importances_
        top_feats = sorted(zip(selected_directional_feats, importances), key=lambda x: -x[1])
        print("\nTop directional features by importance:")
        for feat, imp in top_feats[:10]:
            print(f"  {feat:20} {imp:.4f}")
        # --- END PATCH ---