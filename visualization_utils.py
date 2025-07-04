"""
Enhanced Visualization Utilities for Bitcoin Price Analysis
===========================================================

This module provides comprehensive visualization tools for Bitcoin price data,
technical indicators, model predictions, and backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BitcoinVisualizer:
    """
    Comprehensive visualization toolkit for Bitcoin price analysis.
    """
    
    def __init__(self, style='seaborn', figsize=(12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
    
    def plot_price_analysis(self, data, save_path=None):
        """
        Create comprehensive price analysis charts.
        
        Args:
            data: Bitcoin price data with OHLCV
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle('Bitcoin Price Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Price and volume
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['Close'], linewidth=2, label='Close Price')
        ax1.set_title('Bitcoin Price Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = axes[0, 1]
        ax2.bar(data.index, data['Volume'], alpha=0.7, color='orange')
        ax2.set_title('Trading Volume', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        # Returns distribution
        ax3 = axes[1, 0]
        returns = data['Close'].pct_change().dropna()
        ax3.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.4f}')
        ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Volatility
        ax4 = axes[1, 1]
        volatility = returns.rolling(window=30).std() * np.sqrt(365)
        ax4.plot(volatility.index, volatility, linewidth=2, color='red')
        ax4.set_title('30-Day Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Volatility')
        ax4.grid(True, alpha=0.3)
        
        # Moving averages
        ax5 = axes[2, 0]
        ax5.plot(data.index, data['Close'], linewidth=2, label='Close', alpha=0.7)
        if 'SMA7' in data.columns:
            ax5.plot(data.index, data['SMA7'], linewidth=1, label='SMA7')
        if 'SMA25' in data.columns:
            ax5.plot(data.index, data['SMA25'], linewidth=1, label='SMA25')
        if 'SMA50' in data.columns:
            ax5.plot(data.index, data['SMA50'], linewidth=1, label='SMA50')
        ax5.set_title('Price with Moving Averages', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Price (USD)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # RSI
        ax6 = axes[2, 1]
        if 'RSI' in data.columns:
            ax6.plot(data.index, data['RSI'], linewidth=2, color='purple')
            ax6.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax6.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax6.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
        ax6.set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('RSI')
        ax6.set_ylim(0, 100)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # MACD
        ax7 = axes[3, 0]
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            ax7.plot(data.index, data['MACD'], linewidth=2, label='MACD', color='blue')
            ax7.plot(data.index, data['MACD_Signal'], linewidth=2, label='Signal', color='red')
            if 'MACD_Hist' in data.columns:
                ax7.bar(data.index, data['MACD_Hist'], alpha=0.7, label='Histogram', color='gray')
        ax7.set_title('MACD Indicator', fontsize=14, fontweight='bold')
        ax7.set_ylabel('MACD')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Price vs Volume correlation
        ax8 = axes[3, 1]
        ax8.scatter(data['Volume'], data['Close'].pct_change(), alpha=0.6)
        ax8.set_title('Volume vs Price Change Correlation', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Volume')
        ax8.set_ylabel('Price Change')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_interactive(self, data, predictions, confidence_intervals=None):
        """
        Create interactive prediction plots using Plotly.
        
        Args:
            data: Historical price data
            predictions: Future predictions DataFrame
            confidence_intervals: Optional confidence intervals
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price Predictions', 'Prediction Confidence', 
                          'Returns Forecast', 'Volatility Analysis',
                          'Prediction Accuracy', 'Risk Metrics'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Historical price
        fig.add_trace(
            go.Scatter(x=data.index[-100:], y=data['Close'].tail(100),
                      mode='lines', name='Historical Price',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Predictions
        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=predictions['Date'], y=predictions['Predicted_Price'],
                          mode='lines', name='Predicted Price',
                          line=dict(color='orange', width=3)),
                row=1, col=1
            )
            
            # Confidence intervals
            if 'Lower_Price' in predictions.columns and 'Upper_Price' in predictions.columns:
                fig.add_trace(
                    go.Scatter(x=predictions['Date'], y=predictions['Upper_Price'],
                              mode='lines', line=dict(color='rgba(0,0,0,0)'),
                              showlegend=False, name='Upper Bound'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=predictions['Date'], y=predictions['Lower_Price'],
                              mode='lines', fill='tonexty', 
                              fillcolor='rgba(255,165,0,0.2)',
                              line=dict(color='rgba(0,0,0,0)'),
                              name='Confidence Interval'),
                    row=1, col=1
                )
        
        # Confidence interval width
        if predictions is not None and 'Lower_Price' in predictions.columns:
            confidence_width = (predictions['Upper_Price'] - predictions['Lower_Price']) / predictions['Predicted_Price']
            fig.add_trace(
                go.Scatter(x=predictions['Date'], y=confidence_width,
                          mode='lines', name='Confidence Width',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )
        
        # Returns forecast
        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=predictions['Date'], y=predictions['Predicted_Returns'],
                          mode='lines', name='Predicted Returns',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Historical volatility
        if len(data) > 30:
            volatility = data['Close'].pct_change().rolling(30).std() * np.sqrt(365)
            fig.add_trace(
                go.Scatter(x=data.index[-100:], y=volatility.tail(100),
                          mode='lines', name='Historical Volatility',
                          line=dict(color='purple', width=2)),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Bitcoin Price Predictions - Interactive Dashboard',
            height=1000,
            showlegend=True
        )
        
        fig.show()
    
    def plot_model_comparison(self, results_dict, metrics=['RMSE', 'MAPE', 'Directional_Accuracy']):
        """
        Compare performance of different models.
        
        Args:
            results_dict: Dictionary with model names as keys and metrics as values
            metrics: List of metrics to compare
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        models = list(results_dict.keys())
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7, 
                              color=sns.color_palette("husl", len(models)))
            axes[i].set_title(f'{metric} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance_values, top_n=20):
        """
        Plot feature importance from ML models.
        
        Args:
            feature_names: List of feature names
            importance_values: Corresponding importance values
            top_n: Number of top features to display
        """
        # Create DataFrame and sort by importance
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(df)), df['Importance'], alpha=0.8)
        plt.yticks(range(len(df)), df['Feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        
        # Color bars based on importance
        colors = plt.cm.viridis(df['Importance'] / df['Importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, data, features=None):
        """
        Plot correlation heatmap of features.
        
        Args:
            data: DataFrame with features
            features: List of specific features to include
        """
        if features:
            corr_data = data[features]
        else:
            corr_data = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, 
                   annot=False, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_regime_analysis(self, data):
        """
        Visualize market regime analysis.
        
        Args:
            data: DataFrame with regime information
        """
        if 'Regime' not in data.columns:
            print("No regime data available")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Price with regime coloring
        ax1 = axes[0]
        regime_colors = {0: 'gray', 1: 'green', 2: 'red', 3: 'blue'}
        
        for regime in data['Regime'].unique():
            regime_data = data[data['Regime'] == regime]
            ax1.scatter(regime_data.index, regime_data['Close'], 
                       c=regime_colors.get(regime, 'black'), 
                       label=f'Regime {regime}', alpha=0.6, s=1)
        
        ax1.set_title('Bitcoin Price Colored by Market Regime', fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regime distribution over time
        ax2 = axes[1]
        regime_counts = data.groupby([data.index.year, 'Regime']).size().unstack(fill_value=0)
        regime_counts.plot(kind='bar', stacked=True, ax=ax2, 
                          color=[regime_colors.get(i, 'black') for i in regime_counts.columns])
        ax2.set_title('Regime Distribution by Year', fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Days')
        ax2.legend(title='Regime')
        
        # Regime statistics
        ax3 = axes[2]
        regime_stats = data.groupby('Regime')['Close'].pct_change().agg(['mean', 'std'])
        x = np.arange(len(regime_stats))
        width = 0.35
        
        ax3.bar(x - width/2, regime_stats['mean'], width, label='Mean Return', alpha=0.8)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, regime_stats['std'], width, label='Volatility', 
                    alpha=0.8, color='orange')
        
        ax3.set_title('Regime Return and Volatility Statistics', fontweight='bold')
        ax3.set_xlabel('Regime')
        ax3.set_ylabel('Mean Return')
        ax3_twin.set_ylabel('Volatility')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Regime {i}' for i in regime_stats.index])
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_risk_metrics(self, returns, benchmark_returns=None):
        """
        Plot comprehensive risk metrics.
        
        Args:
            returns: Portfolio/strategy returns
            benchmark_returns: Benchmark returns for comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Returns distribution
        ax1 = axes[0, 0]
        ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax2 = axes[0, 1]
        cumulative_returns = (1 + returns).cumprod()
        ax2.plot(cumulative_returns.index, cumulative_returns, linewidth=2, label='Strategy')
        if benchmark_returns is not None:
            cumulative_benchmark = (1 + benchmark_returns).cumprod()
            ax2.plot(cumulative_benchmark.index, cumulative_benchmark, 
                    linewidth=2, label='Benchmark', alpha=0.7)
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Drawdown
        ax3 = axes[0, 2]
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red')
        ax3.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # Rolling volatility
        ax4 = axes[1, 0]
        rolling_vol = returns.rolling(30).std() * np.sqrt(365)
        ax4.plot(rolling_vol.index, rolling_vol, linewidth=2, color='purple')
        ax4.set_title('30-Day Rolling Volatility')
        ax4.set_ylabel('Annualized Volatility')
        ax4.grid(True, alpha=0.3)
        
        # Risk-return scatter
        ax5 = axes[1, 1]
        if benchmark_returns is not None:
            rolling_returns = returns.rolling(60).mean() * 365
            rolling_vol = returns.rolling(60).std() * np.sqrt(365)
            ax5.scatter(rolling_vol, rolling_returns, alpha=0.6, color='blue')
            ax5.set_xlabel('Volatility (Annualized)')
            ax5.set_ylabel('Return (Annualized)')
            ax5.set_title('Risk-Return Profile')
            ax5.grid(True, alpha=0.3)
        
        # VaR analysis
        ax6 = axes[1, 2]
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        ax6.hist(returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax6.axvline(var_95, color='orange', linestyle='--', 
                   label=f'95% VaR: {var_95:.4f}')
        ax6.axvline(var_99, color='red', linestyle='--', 
                   label=f'99% VaR: {var_99:.4f}')
        ax6.set_title('Value at Risk (VaR)')
        ax6.set_xlabel('Returns')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_dashboard_html(self, data, predictions=None, save_path='bitcoin_dashboard.html'):
        """
        Create an interactive HTML dashboard.
        
        Args:
            data: Bitcoin price data
            predictions: Prediction results
            save_path: Path to save HTML file
        """
        from plotly.offline import plot
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('Price Chart', 'Volume', 'Technical Indicators', 
                          'Predictions', 'Returns', 'Volatility', 
                          'Correlation', 'Performance'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces for each subplot
        # Price and volume
        fig.add_trace(
            go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                          low=data['Low'], close=data['Close'], name='OHLC'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', opacity=0.7),
            row=1, col=2
        )
        
        # Technical indicators
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
        
        # Predictions
        if predictions is not None:
            fig.add_trace(
                go.Scatter(x=predictions['Date'], y=predictions['Predicted_Price'],
                          name='Predictions', line=dict(color='orange')),
                row=2, col=2
            )
        
        # Returns
        returns = data['Close'].pct_change()
        fig.add_trace(
            go.Histogram(x=returns, name='Returns Distribution', nbinsx=50),
            row=3, col=1
        )
        
        # Volatility
        volatility = returns.rolling(30).std() * np.sqrt(365)
        fig.add_trace(
            go.Scatter(x=data.index, y=volatility, name='Volatility', line=dict(color='red')),
            row=3, col=2
        )
        
        fig.update_layout(
            title='Bitcoin Analysis Dashboard',
            height=1200,
            showlegend=False
        )
        
        # Save to HTML
        plot(fig, filename=save_path, auto_open=False)
        print(f"Dashboard saved to {save_path}")

# Example usage functions
def demo_visualizations():
    """Demonstrate the visualization capabilities."""
    # Generate sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    price = 30000
    prices = [price]
    volumes = []
    
    for i in range(999):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
        volumes.append(np.random.exponential(1000000))
    
    # Create sample DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': np.array(prices) * np.random.uniform(0.98, 1.02, 1000),
        'High': np.array(prices) * np.random.uniform(1.0, 1.05, 1000),
        'Low': np.array(prices) * np.random.uniform(0.95, 1.0, 1000),
        'Volume': volumes
    }).set_index('Date')
    
    # Add some technical indicators
    data['SMA7'] = data['Close'].rolling(7).mean()
    data['SMA25'] = data['Close'].rolling(25).mean()
    data['RSI'] = np.random.uniform(20, 80, 1000)
    data['Regime'] = np.random.choice([0, 1, 2, 3], 1000)
    
    # Initialize visualizer
    viz = BitcoinVisualizer()
    
    # Create visualizations
    print("Creating price analysis dashboard...")
    viz.plot_price_analysis(data)
    
    print("Creating regime analysis...")
    viz.plot_regime_analysis(data)
    
    print("Creating risk metrics...")
    returns = data['Close'].pct_change().dropna()
    viz.plot_risk_metrics(returns)

if __name__ == "__main__":
    demo_visualizations()