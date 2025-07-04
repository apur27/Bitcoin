import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our Bitcoin forecaster
from bitcoinML7 import UltimateBitcoinForecaster

# Configure Streamlit page
st.set_page_config(
    page_title="🚀 Bitcoin Price Forecaster",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = UltimateBitcoinForecaster()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚀 Bitcoin Price Forecaster</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys section
        st.subheader("🔑 API Keys")
        twelvedata_key = st.text_input("TwelveData API Key", type="password", 
                                      help="Get free API key from twelvedata.com")
        fred_key = st.text_input("FRED API Key", type="password",
                                help="Get free API key from fred.stlouisfed.org")
        
        # Data parameters
        st.subheader("📊 Data Parameters")
        days_data = st.slider("Days of Historical Data", 500, 5000, 2000, 
                             help="More data = better training but slower processing")
        
        # Model parameters
        st.subheader("🤖 Model Parameters")
        lookback_days = st.slider("Lookback Period (days)", 10, 60, 30,
                                 help="Number of days to look back for predictions")
        prediction_days = st.slider("Prediction Horizon (days)", 1, 90, 30,
                                   help="Number of days to predict ahead")
        test_size = st.slider("Test Set Size", 0.05, 0.3, 0.1,
                             help="Fraction of data for testing")
        
        # Action buttons
        st.subheader("🎯 Actions")
        load_data_btn = st.button("📈 Load Data", use_container_width=True)
        train_models_btn = st.button("🧠 Train Models", use_container_width=True, 
                                    disabled=not st.session_state.data_loaded)
        predict_btn = st.button("🔮 Generate Predictions", use_container_width=True,
                               disabled=not st.session_state.models_trained)

    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Data Analysis", "🧠 Model Training", "🔮 Predictions", "📋 Documentation"])
    
    with tab1:
        dashboard_tab(load_data_btn, train_models_btn, predict_btn, 
                     twelvedata_key, fred_key, days_data, lookback_days, 
                     prediction_days, test_size)
    
    with tab2:
        data_analysis_tab()
    
    with tab3:
        model_training_tab()
    
    with tab4:
        predictions_tab()
        
    with tab5:
        documentation_tab()

def dashboard_tab(load_data_btn, train_models_btn, predict_btn, 
                 twelvedata_key, fred_key, days_data, lookback_days, 
                 prediction_days, test_size):
    
    # Data Loading Section
    if load_data_btn:
        if not twelvedata_key:
            st.error("❌ Please provide TwelveData API key")
            return
            
        with st.spinner("📥 Loading Bitcoin data..."):
            st.session_state.forecaster.fred_api_key = fred_key if fred_key else None
            
            # Fetch data
            data = st.session_state.forecaster.fetch_bitcoin_data(days=days_data, api_key=twelvedata_key)
            if data is not None:
                st.session_state.forecaster.fetch_volume_from_binance(days=days_data)
                st.session_state.forecaster.add_cycle_features()
                st.session_state.forecaster.calculate_indicators()
                st.session_state.forecaster.engineer_directional_features()
                st.session_state.forecaster.detect_market_regimes()
                st.session_state.data_loaded = True
                st.success("✅ Data loaded successfully!")
            else:
                st.error("❌ Failed to load data. Check your API key.")

    # Model Training Section
    if train_models_btn and st.session_state.data_loaded:
        with st.spinner("🧠 Training models... This may take several minutes."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preparing data...")
                progress_bar.progress(20)
                
                # Prepare data
                ml_data = st.session_state.forecaster.prepare_ml_data(lookback=lookback_days)
                if ml_data is None:
                    st.error("❌ Failed to prepare ML data")
                    return
                
                X_scaled, y_scaled, dates, _ = ml_data
                
                status_text.text("Optimizing hyperparameters...")
                progress_bar.progress(40)
                
                # Optimize hyperparameters
                st.session_state.forecaster.optimize_hyperparameters(X_scaled, y_scaled, dates, lookback_days)
                
                status_text.text("Training ensemble models...")
                progress_bar.progress(80)
                
                # Train models
                results = st.session_state.forecaster.train_all_models(test_size=test_size, lookback=lookback_days)
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                st.session_state.models_trained = True
                st.success("✅ Models trained successfully!")
                
                # Store results
                st.session_state.training_results = results
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

    # Predictions Section
    if predict_btn and st.session_state.models_trained:
        with st.spinner("🔮 Generating predictions..."):
            try:
                predictions = st.session_state.forecaster.predict_with_cycle_awareness(
                    lookback=lookback_days, 
                    days_ahead=prediction_days
                )
                
                if predictions is not None:
                    st.session_state.predictions = predictions
                    st.success("✅ Predictions generated successfully!")
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RMSE", f"{st.session_state.forecaster.final_rmse:.4f}")
                    with col2:
                        st.metric("MAPE", f"{st.session_state.forecaster.final_mape:.4f}")
                    with col3:
                        st.metric("Directional Accuracy", f"{st.session_state.forecaster.final_dir_acc:.4f}")
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

    # Display current status
    st.subheader("📊 System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Loaded" if st.session_state.data_loaded else "❌ Not Loaded"
        st.metric("Data Status", status)
    
    with col2:
        status = "✅ Trained" if st.session_state.models_trained else "❌ Not Trained"
        st.metric("Models Status", status)
    
    with col3:
        status = "✅ Available" if hasattr(st.session_state, 'predictions') else "❌ Not Available"
        st.metric("Predictions Status", status)

def data_analysis_tab():
    st.header("📈 Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Dashboard tab.")
        return
    
    data = st.session_state.forecaster.data
    
    # Basic statistics
    st.subheader("📊 Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Days", len(data))
    with col2:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:,.2f}")
    with col3:
        st.metric("30-Day Change", f"{((data['Close'].iloc[-1] / data['Close'].iloc[-30]) - 1) * 100:.2f}%")
    with col4:
        st.metric("Volatility (30d)", f"{data['Close'].pct_change().tail(30).std() * np.sqrt(365) * 100:.1f}%")
    
    # Price chart
    st.subheader("💰 Bitcoin Price History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Bitcoin Price'))
    fig.update_layout(
        title="Bitcoin Price Over Time",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    st.subheader("📈 Technical Indicators")
    
    # RSI
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(title="RSI (14-day)", yaxis_title="RSI")
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility'))
        fig_vol.update_layout(title="Price Volatility", yaxis_title="Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Market regimes
    if 'Regime' in data.columns:
        st.subheader("🏛️ Market Regimes")
        regime_counts = data['Regime'].value_counts()
        fig_regime = px.pie(values=regime_counts.values, names=regime_counts.index, 
                           title="Market Regime Distribution")
        st.plotly_chart(fig_regime, use_container_width=True)

def model_training_tab():
    st.header("🧠 Model Training Results")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first from the Dashboard tab.")
        return
    
    # Model performance metrics
    st.subheader("📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Root Mean Square Error", f"{st.session_state.forecaster.final_rmse:.6f}")
    with col2:
        st.metric("Mean Absolute Percentage Error", f"{st.session_state.forecaster.final_mape:.4f}")
    with col3:
        st.metric("Directional Accuracy", f"{st.session_state.forecaster.final_dir_acc:.4f}")
    
    # Feature importance (if available)
    if hasattr(st.session_state.forecaster, 'xgb_model') and st.session_state.forecaster.xgb_model:
        st.subheader("🔍 Feature Importance")
        try:
            importance = st.session_state.forecaster.xgb_model.feature_importances_
            features = st.session_state.forecaster.selected_features
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(20)
            
            fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                   orientation='h', title="Top 20 Most Important Features")
            st.plotly_chart(fig_importance, use_container_width=True)
        except:
            st.info("Feature importance not available")

def predictions_tab():
    st.header("🔮 Future Predictions")
    
    if not hasattr(st.session_state, 'predictions'):
        st.warning("⚠️ Please generate predictions first from the Dashboard tab.")
        return
    
    predictions = st.session_state.predictions
    
    # Risk disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Risk Disclaimer:</strong> These predictions are for educational purposes only. 
        Cryptocurrency markets are highly volatile and unpredictable. Never invest more than you can afford to lose.
    </div>
    """, unsafe_allow_html=True)
    
    # Predictions table
    st.subheader("📊 Prediction Results")
    st.dataframe(predictions[['Date', 'Predicted_Returns', 'Predicted_Price', 'Lower_Price', 'Upper_Price']], 
                use_container_width=True)
    
    # Predictions chart
    st.subheader("📈 Price Predictions with Confidence Intervals")
    
    fig = go.Figure()
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=predictions['Date'], 
        y=predictions['Predicted_Price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='orange', width=3)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Upper_Price'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=predictions['Lower_Price'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Interval'
    ))
    
    # Add historical price
    if st.session_state.data_loaded:
        recent_data = st.session_state.forecaster.data.tail(100)
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title="Bitcoin Price Predictions with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_return = predictions['Predicted_Returns'].mean()
        st.metric("Average Daily Return", f"{avg_return:.4f}")
    
    with col2:
        total_return = (predictions['Predicted_Price'].iloc[-1] / predictions['Predicted_Price'].iloc[0] - 1) * 100
        st.metric("Total Predicted Return", f"{total_return:.2f}%")
    
    with col3:
        volatility = predictions['Predicted_Returns'].std() * np.sqrt(365)
        st.metric("Annualized Volatility", f"{volatility:.2f}")
    
    with col4:
        max_price = predictions['Predicted_Price'].max()
        st.metric("Predicted Max Price", f"${max_price:,.2f}")

def documentation_tab():
    st.header("📋 Documentation")
    
    st.markdown("""
    ## 🚀 How to Use This Dashboard
    
    ### 1. Configuration
    - Enter your API keys in the sidebar
    - Adjust data and model parameters
    - Choose prediction horizon
    
    ### 2. Load Data
    - Click "📈 Load Data" to fetch Bitcoin price data
    - System will automatically add technical indicators and external data
    
    ### 3. Train Models
    - Click "🧠 Train Models" to train the ensemble of ML models
    - This includes XGBoost, LSTM, Prophet, and CatBoost
    - Hyperparameters are automatically optimized
    
    ### 4. Generate Predictions
    - Click "🔮 Generate Predictions" to forecast future prices
    - View results with confidence intervals
    
    ## 🤖 Models Used
    
    - **XGBoost**: Gradient boosting for robust predictions
    - **LSTM**: Deep learning with attention mechanism
    - **Prophet**: Time series forecasting with seasonality
    - **CatBoost**: Advanced gradient boosting
    - **Ensemble**: Weighted combination of all models
    
    ## 📊 Data Sources
    
    - **Price Data**: TwelveData API
    - **Volume**: Binance API
    - **Sentiment**: Fear & Greed Index
    - **Macro**: Federal Reserve Economic Data (FRED)
    - **On-Chain**: Blockchain.com metrics
    
    ## ⚠️ Important Notes
    
    - Predictions are for educational purposes only
    - Cryptocurrency markets are highly volatile
    - Past performance doesn't guarantee future results
    - Always do your own research before investing
    
    ## 🔧 API Keys
    
    ### TwelveData
    1. Visit [twelvedata.com](https://twelvedata.com)
    2. Create free account
    3. Get API key from dashboard
    
    ### FRED (Optional)
    1. Visit [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
    2. Request free API key
    3. Enables macroeconomic data integration
    """)

if __name__ == "__main__":
    main()