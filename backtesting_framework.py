"""
Bitcoin Forecasting Backtesting Framework
==========================================

A comprehensive backtesting system to evaluate the performance of Bitcoin prediction models
on historical data with various trading strategies and risk management techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinBacktester:
    """
    Comprehensive backtesting framework for Bitcoin prediction models.
    
    Features:
    - Walk-forward analysis
    - Multiple trading strategies
    - Risk management
    - Performance metrics
    - Visualization
    """
    
    def __init__(self, initial_capital=10000, transaction_cost=0.001, slippage=0.0005):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
            transaction_cost: Transaction cost as a fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage cost as a fraction
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.results = {}
        self.trades = []
        
    def walk_forward_backtest(self, forecaster, data, window_size=365, step_size=30, 
                             start_date=None, end_date=None):
        """
        Perform walk-forward backtesting.
        
        Args:
            forecaster: Trained Bitcoin forecaster object
            data: Historical price data
            window_size: Training window size in days
            step_size: Step size for moving the window
            start_date: Start date for backtesting
            end_date: End date for backtesting
        """
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = data.index[window_size]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = data.index[-30]  # Leave 30 days at the end
        
        print(f"Starting walk-forward backtest from {start_date} to {end_date}")
        print(f"Window size: {window_size} days, Step size: {step_size} days")
        
        backtest_results = []
        current_date = start_date
        
        while current_date <= end_date:
            # Define training and testing periods
            train_start = current_date - timedelta(days=window_size)
            train_end = current_date
            test_start = current_date + timedelta(days=1)
            test_end = min(current_date + timedelta(days=step_size), end_date)
            
            print(f"Processing: Train {train_start.date()} to {train_end.date()}, "
                  f"Test {test_start.date()} to {test_end.date()}")
            
            try:
                # Get training and testing data
                train_data = data[train_start:train_end]
                test_data = data[test_start:test_end]
                
                if len(train_data) < 100 or len(test_data) < 1:
                    current_date += timedelta(days=step_size)
                    continue
                
                # Simulate model training (in practice, you'd retrain here)
                # For now, we'll use the existing trained model
                
                # Generate predictions for the test period
                predictions = self._generate_predictions(forecaster, test_data)
                
                # Evaluate predictions
                if predictions is not None and len(predictions) > 0:
                    metrics = self._evaluate_predictions(test_data, predictions)
                    metrics['train_start'] = train_start
                    metrics['train_end'] = train_end
                    metrics['test_start'] = test_start
                    metrics['test_end'] = test_end
                    backtest_results.append(metrics)
                
            except Exception as e:
                print(f"Error in period {current_date}: {e}")
            
            current_date += timedelta(days=step_size)
        
        self.backtest_results = pd.DataFrame(backtest_results)
        return self.backtest_results
    
    def _generate_predictions(self, forecaster, test_data):
        """Generate predictions for test data."""
        try:
            # Prepare features for the test period
            # This is a simplified version - in practice, you'd need to
            # ensure proper feature engineering
            predictions = []
            
            for i in range(len(test_data)):
                # Simple momentum-based prediction for demonstration
                recent_returns = test_data['Close'].pct_change().iloc[max(0, i-5):i+1]
                if len(recent_returns) > 0:
                    pred = recent_returns.mean()
                    predictions.append(pred)
                else:
                    predictions.append(0.0)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return None
    
    def _evaluate_predictions(self, test_data, predictions):
        """Evaluate prediction performance."""
        actual_returns = test_data['Close'].pct_change().dropna().values
        
        # Align predictions with actual returns
        min_len = min(len(actual_returns), len(predictions))
        actual_returns = actual_returns[:min_len]
        predictions = predictions[:min_len]
        
        if len(actual_returns) == 0:
            return {}
        
        # Calculate metrics
        mse = np.mean((actual_returns - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_returns - predictions))
        
        # Directional accuracy
        actual_direction = np.sign(actual_returns)
        predicted_direction = np.sign(predictions)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Information coefficient (correlation)
        ic = np.corrcoef(actual_returns, predictions)[0, 1] if len(actual_returns) > 1 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'information_coefficient': ic,
            'n_predictions': len(predictions)
        }
    
    def strategy_backtest(self, data, predictions, strategy='momentum', 
                         stop_loss=0.05, take_profit=0.15):
        """
        Backtest trading strategies based on predictions.
        
        Args:
            data: Price data
            predictions: Model predictions
            strategy: Trading strategy ('momentum', 'mean_reversion', 'threshold')
            stop_loss: Stop loss threshold
            take_profit: Take profit threshold
        """
        portfolio_value = [self.initial_capital]
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        cash = self.initial_capital
        bitcoin_holdings = 0
        
        trades = []
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            prediction = predictions[i] if i < len(predictions) else 0
            
            # Generate trading signals based on strategy
            signal = self._generate_signal(strategy, prediction, data.iloc[:i+1])
            
            # Execute trades
            if position == 0:  # No current position
                if signal > 0:  # Buy signal
                    # Calculate position size (risk management)
                    position_size = self._calculate_position_size(cash, current_price)
                    bitcoin_holdings = position_size / current_price
                    cash -= position_size * (1 + self.transaction_cost + self.slippage)
                    position = 1
                    entry_price = current_price
                    
                    trades.append({
                        'date': data.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': bitcoin_holdings,
                        'value': position_size
                    })
                    
                elif signal < 0:  # Sell signal (short)
                    # For simplicity, we'll only implement long positions
                    pass
                    
            elif position == 1:  # Currently long
                # Check exit conditions
                pnl_pct = (current_price - entry_price) / entry_price
                
                should_exit = (
                    signal < 0 or  # Sell signal
                    pnl_pct <= -stop_loss or  # Stop loss
                    pnl_pct >= take_profit  # Take profit
                )
                
                if should_exit:
                    # Sell position
                    sell_value = bitcoin_holdings * current_price * (1 - self.transaction_cost - self.slippage)
                    cash += sell_value
                    
                    trades.append({
                        'date': data.index[i],
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': bitcoin_holdings,
                        'value': sell_value,
                        'pnl': sell_value - (bitcoin_holdings * entry_price)
                    })
                    
                    bitcoin_holdings = 0
                    position = 0
                    entry_price = 0
            
            # Calculate current portfolio value
            current_value = cash + (bitcoin_holdings * current_price if bitcoin_holdings > 0 else 0)
            portfolio_value.append(current_value)
        
        self.trades = trades
        self.portfolio_values = portfolio_value
        
        return self._calculate_strategy_metrics(data, portfolio_value)
    
    def _generate_signal(self, strategy, prediction, historical_data):
        """Generate trading signals based on strategy."""
        if strategy == 'momentum':
            # Simple momentum strategy
            threshold = 0.01  # 1% threshold
            if prediction > threshold:
                return 1  # Buy
            elif prediction < -threshold:
                return -1  # Sell
            else:
                return 0  # Hold
                
        elif strategy == 'mean_reversion':
            # Mean reversion strategy
            recent_returns = historical_data['Close'].pct_change().tail(5).mean()
            if recent_returns > 0.02 and prediction < 0:
                return -1  # Sell (expect reversion)
            elif recent_returns < -0.02 and prediction > 0:
                return 1  # Buy (expect reversion)
            else:
                return 0
                
        elif strategy == 'threshold':
            # Threshold-based strategy
            if prediction > 0.005:  # 0.5% threshold
                return 1
            elif prediction < -0.005:
                return -1
            else:
                return 0
        
        return 0
    
    def _calculate_position_size(self, cash, price):
        """Calculate position size based on available cash and risk management."""
        max_position_value = cash * 0.1  # Maximum 10% of portfolio per trade
        return min(max_position_value, cash * 0.95)  # Keep 5% cash buffer
    
    def _calculate_strategy_metrics(self, data, portfolio_values):
        """Calculate comprehensive strategy performance metrics."""
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        benchmark_returns = data['Close'].pct_change().dropna()
        
        # Align the series
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        benchmark_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
        
        annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (365 / len(portfolio_values)) - 1) * 100
        volatility = portfolio_returns.std() * np.sqrt(365) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_portfolio_value': portfolio_values[-1]
        }
    
    def plot_backtest_results(self, data):
        """Plot comprehensive backtest results."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Bitcoin Strategy Backtesting Results', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(self.portfolio_values, label='Strategy', color='blue')
        axes[0, 0].plot([self.initial_capital * (data['Close'].iloc[i] / data['Close'].iloc[0]) 
                        for i in range(len(data))], label='Buy & Hold', color='orange')
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - peak) / peak * 100
        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(drawdown, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Trade distribution
        if self.trades:
            pnls = [t.get('pnl', 0) for t in self.trades if 'pnl' in t]
            if pnls:
                axes[1, 0].hist(pnls, bins=20, alpha=0.7, color='green')
                axes[1, 0].axvline(0, color='red', linestyle='--')
                axes[1, 0].set_title('Trade P&L Distribution')
                axes[1, 0].set_xlabel('P&L ($)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
        
        # Monthly returns heatmap
        portfolio_returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if len(portfolio_returns) > 30:
            try:
                dates = pd.date_range(start=data.index[0], periods=len(portfolio_returns), freq='D')
                monthly_returns = portfolio_returns.groupby([dates.year, dates.month]).apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100
                
                # Create a pivot table for the heatmap
                monthly_data = monthly_returns.reset_index()
                monthly_data.columns = ['Year', 'Month', 'Return']
                pivot = monthly_data.pivot(index='Year', columns='Month', values='Return')
                
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                           ax=axes[1, 1], cbar_kws={'label': 'Return (%)'})
                axes[1, 1].set_title('Monthly Returns Heatmap')
            except:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data for heatmap', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Risk-return scatter
        if hasattr(self, 'backtest_results') and len(self.backtest_results) > 0:
            axes[2, 0].scatter(self.backtest_results['rmse'], self.backtest_results['directional_accuracy'])
            axes[2, 0].set_xlabel('RMSE')
            axes[2, 0].set_ylabel('Directional Accuracy')
            axes[2, 0].set_title('Risk vs Accuracy')
            axes[2, 0].grid(True)
        
        # Performance metrics text
        if hasattr(self, 'strategy_metrics'):
            metrics_text = f"""
            Total Return: {self.strategy_metrics['total_return']:.2f}%
            Annual Return: {self.strategy_metrics['annual_return']:.2f}%
            Volatility: {self.strategy_metrics['volatility']:.2f}%
            Sharpe Ratio: {self.strategy_metrics['sharpe_ratio']:.2f}
            Max Drawdown: {self.strategy_metrics['max_drawdown']:.2f}%
            Win Rate: {self.strategy_metrics['win_rate']:.1f}%
            Total Trades: {self.strategy_metrics['total_trades']}
            """
            axes[2, 1].text(0.1, 0.9, metrics_text, transform=axes[2, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace')
            axes[2, 1].set_title('Performance Metrics')
            axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_backtest_report(self):
        """Generate a comprehensive backtest report."""
        report = {
            'strategy_metrics': getattr(self, 'strategy_metrics', {}),
            'trade_summary': {
                'total_trades': len(self.trades),
                'profitable_trades': len([t for t in self.trades if t.get('pnl', 0) > 0]),
                'average_trade_pnl': np.mean([t.get('pnl', 0) for t in self.trades if 'pnl' in t]) if self.trades else 0,
                'largest_win': max([t.get('pnl', 0) for t in self.trades if 'pnl' in t], default=0),
                'largest_loss': min([t.get('pnl', 0) for t in self.trades if 'pnl' in t], default=0)
            }
        }
        
        if hasattr(self, 'backtest_results'):
            report['prediction_metrics'] = {
                'average_rmse': self.backtest_results['rmse'].mean(),
                'average_directional_accuracy': self.backtest_results['directional_accuracy'].mean(),
                'average_information_coefficient': self.backtest_results['information_coefficient'].mean()
            }
        
        return report

# Example usage
def run_backtest_example():
    """Example of how to use the backtesting framework."""
    from bitcoinML7 import UltimateBitcoinForecaster
    
    # Initialize forecaster and backtester
    forecaster = UltimateBitcoinForecaster()
    backtester = BitcoinBacktester(initial_capital=10000)
    
    # Fetch and prepare data
    forecaster.fetch_bitcoin_data(days=1000, api_key='your_api_key')
    if forecaster.data is not None:
        forecaster.calculate_indicators()
        
        # Run walk-forward backtest
        backtest_results = backtester.walk_forward_backtest(
            forecaster, 
            forecaster.data,
            window_size=365,
            step_size=30
        )
        
        # Generate dummy predictions for strategy backtest
        predictions = np.random.normal(0, 0.02, len(forecaster.data))
        
        # Run strategy backtest
        strategy_metrics = backtester.strategy_backtest(
            forecaster.data, 
            predictions, 
            strategy='momentum'
        )
        
        # Plot results
        backtester.plot_backtest_results(forecaster.data)
        
        # Generate report
        report = backtester.generate_backtest_report()
        print("Backtest Report:")
        print(report)

if __name__ == "__main__":
    run_backtest_example()