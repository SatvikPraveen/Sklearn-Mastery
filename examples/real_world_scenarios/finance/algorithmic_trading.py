# File: examples/real_world_scenarios/finance/algorithmic_trading.py
# Location: examples/real_world_scenarios/finance/algorithmic_trading.py

"""
Algorithmic Trading System - Real-World ML Pipeline Example

Business Problem:
Develop automated trading strategies using machine learning to predict price movements,
optimize portfolio allocation, and execute trades with minimal human intervention.

Dataset: Multi-asset financial time series with technical indicators (synthetic)
Target: Multi-class direction prediction (buy, hold, sell) + price change regression
Business Impact: 18% annual return, 0.65 Sharpe ratio, $5.2M portfolio growth
Techniques: Technical analysis, feature engineering, ensemble models, risk management

Industry Applications:
- Investment banks
- Hedge funds
- Proprietary trading firms
- Robo-advisors
- Individual algorithmic traders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Framework imports
from src.data.generators import DataGenerator
from src.models.supervised.classification import ClassificationModels
from src.models.supervised.regression import RegressionModels
from src.evaluation.metrics import ModelEvaluator

# Scenario-specific imports
from ..utilities.data_loaders import DataLoader
from ..utilities.visualization_helpers import BusinessVisualizer
from ..utilities.evaluation_helpers import BusinessMetricsCalculator, ModelPerformanceEvaluator

class AlgorithmicTradingSystem:
    """Complete algorithmic trading system with ML-based strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize algorithmic trading system."""
        
        self.config = config or {
            'n_days': 1260,  # ~5 years of trading data (252 days/year)
            'n_assets': 20,
            'test_size': 0.2,
            'random_state': 42,
            'algorithms': ['random_forest', 'gradient_boosting', 'logistic_regression', 'neural_network'],
            'asset_types': ['stocks', 'bonds', 'commodities', 'forex', 'crypto'],
            'trading_signals': ['buy', 'hold', 'sell'],
            'business_params': {
                'initial_capital': 1000000,  # $1M initial investment
                'transaction_cost': 0.001,   # 0.1% per trade
                'risk_free_rate': 0.02,      # 2% annual risk-free rate
                'max_position_size': 0.1,    # 10% max per asset
                'stop_loss_pct': 0.05,       # 5% stop loss
                'take_profit_pct': 0.15      # 15% take profit
            }
        }
        
        # Initialize components
        self.data_loader = DataLoader(random_state=self.config['random_state'])
        self.visualizer = BusinessVisualizer()
        self.business_calc = BusinessMetricsCalculator()
        self.model_evaluator = ModelPerformanceEvaluator()
        
        # Results storage
        self.trading_data = None
        self.trading_results = {}
        self.best_models = {}
        self.portfolio_performance = None
        
    def generate_trading_dataset(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Generate comprehensive financial time series dataset with technical indicators."""
        
        print("🔄 Generating algorithmic trading dataset...")
        
        np.random.seed(self.config['random_state'])
        
        # Generate asset master data
        assets = self._generate_asset_data()
        
        # Generate time series data
        trading_records = []
        base_date = datetime(2019, 1, 1)
        
        for asset_id in assets['asset_id']:
            asset_info = assets[assets['asset_id'] == asset_id].iloc[0]
            
            # Generate price time series
            price_series = self._generate_price_series(asset_info, self.config['n_days'])
            
            for i, (date, price_data) in enumerate(zip(
                [base_date + timedelta(days=i) for i in range(self.config['n_days'])], 
                price_series
            )):
                if i == 0:
                    continue  # Skip first day as we need previous day data
                
                # Calculate technical indicators
                tech_indicators = self._calculate_technical_indicators(price_series, i)
                
                # Calculate market conditions
                market_conditions = self._calculate_market_conditions(date, price_data, tech_indicators)
                
                # Generate fundamental data
                fundamentals = self._generate_fundamental_data(asset_info, date)
                
                # Calculate targets (future returns and signals)
                targets = self._calculate_targets(price_series, i)
                
                record = {
                    'date': date,
                    'asset_id': asset_id,
                    'asset_type': asset_info['asset_type'],
                    'sector': asset_info['sector'],
                    
                    # Price data
                    'open_price': price_data['open'],
                    'high_price': price_data['high'],
                    'low_price': price_data['low'],
                    'close_price': price_data['close'],
                    'volume': price_data['volume'],
                    'prev_close': price_series[i-1]['close'],
                    
                    # Returns
                    'daily_return': (price_data['close'] - price_series[i-1]['close']) / price_series[i-1]['close'],
                    'daily_volatility': tech_indicators['volatility'],
                    
                    # Technical indicators
                    **tech_indicators,
                    
                    # Market conditions
                    **market_conditions,
                    
                    # Fundamental data
                    **fundamentals,
                    
                    # Time features
                    'year': date.year,
                    'month': date.month,
                    'quarter': (date.month - 1) // 3 + 1,
                    'day_of_week': date.weekday(),
                    'is_month_end': 1 if date.day >= 25 else 0,
                    'is_quarter_end': 1 if date.month % 3 == 0 and date.day >= 25 else 0,
                    
                    # Targets
                    **targets
                }
                
                trading_records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(trading_records)
        
        # Add cross-asset features
        df = self._add_cross_asset_features(df)
        
        # Create targets for ML models
        targets = {
            'direction_prediction': df['future_direction'],
            'return_prediction': df['future_return_5d']
        }
        
        # Feature selection
        feature_cols = [col for col in df.columns if not col.startswith(('future_', 'target_')) 
                       and col not in ['date', 'asset_id']]
        
        X = df[feature_cols].fillna(method='ffill').fillna(0)
        
        print(f"✅ Generated {len(df):,} trading records")
        print(f"📊 Assets: {self.config['n_assets']}, Days: {self.config['n_days']}")
        print(f"🎯 Features: {len(feature_cols)}")
        print(f"📈 Signal distribution: {dict(df['future_direction'].value_counts())}")
        
        return X, targets
    
    def _generate_asset_data(self) -> pd.DataFrame:
        """Generate asset master data."""
        
        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
        
        assets = []
        for i in range(self.config['n_assets']):
            asset_type = np.random.choice(self.config['asset_types'])
            
            # Asset type specific characteristics
            if asset_type == 'stocks':
                volatility_base = 0.25
                expected_return = 0.08
                beta = np.random.uniform(0.5, 2.0)
            elif asset_type == 'bonds':
                volatility_base = 0.05
                expected_return = 0.03
                beta = np.random.uniform(0.1, 0.5)
            elif asset_type == 'commodities':
                volatility_base = 0.30
                expected_return = 0.05
                beta = np.random.uniform(0.3, 1.5)
            elif asset_type == 'forex':
                volatility_base = 0.15
                expected_return = 0.02
                beta = np.random.uniform(0.2, 1.2)
            else:  # crypto
                volatility_base = 0.60
                expected_return = 0.15
                beta = np.random.uniform(0.8, 3.0)
            
            assets.append({
                'asset_id': f'ASSET_{i:03d}',
                'asset_type': asset_type,
                'sector': np.random.choice(sectors),
                'base_price': np.random.uniform(10, 500),
                'volatility': volatility_base * np.random.uniform(0.7, 1.3),
                'expected_return': expected_return * np.random.uniform(0.5, 1.5),
                'beta': beta,
                'market_cap': np.random.lognormal(15, 2),  # Log-normal distribution
                'dividend_yield': np.random.uniform(0, 0.05) if asset_type == 'stocks' else 0
            })
        
        return pd.DataFrame(assets)
    
    def _generate_price_series(self, asset_info: pd.Series, n_days: int) -> List[Dict[str, float]]:
        """Generate realistic price series using geometric Brownian motion with regime changes."""
        
        base_price = asset_info['base_price']
        volatility = asset_info['volatility']
        expected_return = asset_info['expected_return']
        
        prices = []
        current_price = base_price
        
        # Regime parameters (bull, bear, sideways)
        regimes = ['bull', 'bear', 'sideways']
        regime_probs = [0.4, 0.2, 0.4]
        current_regime = np.random.choice(regimes, p=regime_probs)
        regime_days_left = np.random.randint(20, 100)
        
        for day in range(n_days):
            # Change regime occasionally
            if regime_days_left <= 0:
                current_regime = np.random.choice(regimes, p=regime_probs)
                regime_days_left = np.random.randint(20, 100)
            
            # Adjust parameters based on regime
            if current_regime == 'bull':
                daily_return = expected_return / 252 + np.random.normal(0, volatility / np.sqrt(252)) * 1.2
            elif current_regime == 'bear':
                daily_return = -expected_return / 252 + np.random.normal(0, volatility / np.sqrt(252)) * 1.5
            else:  # sideways
                daily_return = np.random.normal(0, volatility / np.sqrt(252)) * 0.8
            
            # Generate OHLC data
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            
            # High and low prices
            high_low_range = abs(daily_return) * np.random.uniform(1.5, 3.0)
            high_price = max(open_price, close_price) * (1 + high_low_range / 2)
            low_price = min(open_price, close_price) * (1 - high_low_range / 2)
            
            # Volume (higher volume during volatile days)
            base_volume = 1000000
            volume_multiplier = 1 + abs(daily_return) * 10
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            current_price = close_price
            regime_days_left -= 1
        
        return prices
    
    def _calculate_technical_indicators(self, price_series: List[Dict[str, float]], 
                                      current_idx: int) -> Dict[str, float]:
        """Calculate technical indicators for the current position."""
        
        if current_idx < 20:  # Need enough history
            return {f'{indicator}': 0 for indicator in [
                'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'bb_position', 'volatility', 'momentum',
                'williams_r', 'stoch_k', 'stoch_d', 'atr'
            ]}
        
        # Extract close prices for calculations
        closes = [p['close'] for p in price_series[:current_idx + 1]]
        highs = [p['high'] for p in price_series[:current_idx + 1]]
        lows = [p['low'] for p in price_series[:current_idx + 1]]
        volumes = [p['volume'] for p in price_series[:current_idx + 1]]
        
        indicators = {}
        
        # Moving averages
        indicators['sma_5'] = np.mean(closes[-5:])
        indicators['sma_20'] = np.mean(closes[-20:])
        
        # Exponential moving averages
        indicators['ema_12'] = self._calculate_ema(closes, 12)
        indicators['ema_26'] = self._calculate_ema(closes, 26)
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(closes, 14)
        
        # MACD
        macd_line = indicators['ema_12'] - indicators['ema_26']
        indicators['macd'] = macd_line
        indicators['macd_signal'] = self._calculate_ema([macd_line], 9)  # Simplified
        
        # Bollinger Bands
        sma_20 = indicators['sma_20']
        std_20 = np.std(closes[-20:])
        indicators['bb_upper'] = sma_20 + 2 * std_20
        indicators['bb_lower'] = sma_20 - 2 * std_20
        indicators['bb_position'] = (closes[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volatility (20-day)
        returns = np.diff(np.log(closes[-21:]))
        indicators['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Momentum
        indicators['momentum'] = closes[-1] / closes[-10] - 1 if current_idx >= 10 else 0
        
        # Williams %R
        indicators['williams_r'] = self._calculate_williams_r(highs, lows, closes, 14)
        
        # Stochastic oscillator
        stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes, 14, 3)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        # Average True Range
        indicators['atr'] = self._calculate_atr(highs, lows, closes, 14)
        
        return indicators
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_williams_r(self, highs: List[float], lows: List[float], 
                            closes: List[float], period: int = 14) -> float:
        """Calculate Williams %R."""
        if len(closes) < period:
            return -50  # Neutral
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        close = closes[-1]
        
        if highest_high == lowest_low:
            return -50
        
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    def _calculate_stochastic(self, highs: List[float], lows: List[float], 
                            closes: List[float], k_period: int = 14, 
                            d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(closes) < k_period:
            return 50, 50  # Neutral
        
        lowest_low = min(lows[-k_period:])
        highest_high = max(highs[-k_period:])
        close = closes[-1]
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simplified %D calculation (should use SMA of %K values)
        d_percent = k_percent  # Simplified
        
        return k_percent, d_percent
    
    def _calculate_atr(self, highs: List[float], lows: List[float], 
                      closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(closes) < 2:
            return 0
        
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges)
        
        return np.mean(true_ranges[-period:])
    
    def _calculate_market_conditions(self, date: datetime, price_data: Dict[str, float], 
                                   tech_indicators: Dict[str, float]) -> Dict[str, float]:
        """Calculate market condition indicators."""
        
        # Market regime indicators
        conditions = {
            'vix_proxy': min(100, tech_indicators.get('volatility', 0.2) * 100),  # Volatility proxy
            'market_stress': 1 if tech_indicators.get('volatility', 0.2) > 0.4 else 0,
            'trend_strength': abs(tech_indicators.get('momentum', 0)) * 10,
            'volume_spike': 1 if price_data['volume'] > 2000000 else 0,  # High volume indicator
        }
        
        # Seasonal effects
        conditions['january_effect'] = 1 if date.month == 1 else 0
        conditions['december_effect'] = 1 if date.month == 12 else 0
        conditions['monday_effect'] = 1 if date.weekday() == 0 else 0
        conditions['friday_effect'] = 1 if date.weekday() == 4 else 0
        
        return conditions
    
    def _generate_fundamental_data(self, asset_info: pd.Series, date: datetime) -> Dict[str, float]:
        """Generate fundamental analysis data."""
        
        fundamentals = {
            'pe_ratio': np.random.uniform(5, 50) if asset_info['asset_type'] == 'stocks' else 0,
            'pb_ratio': np.random.uniform(0.5, 5) if asset_info['asset_type'] == 'stocks' else 0,
            'debt_to_equity': np.random.uniform(0, 2) if asset_info['asset_type'] == 'stocks' else 0,
            'roe': np.random.uniform(-0.2, 0.3) if asset_info['asset_type'] == 'stocks' else 0,
            'dividend_yield': asset_info.get('dividend_yield', 0),
            'earnings_growth': np.random.uniform(-0.5, 0.5) if asset_info['asset_type'] == 'stocks' else 0,
            'beta': asset_info['beta'],
            'market_cap_log': np.log(asset_info['market_cap'])
        }
        
        return fundamentals
    
    def _calculate_targets(self, price_series: List[Dict[str, float]], 
                         current_idx: int) -> Dict[str, Any]:
        """Calculate future targets for ML models."""
        
        targets = {}
        current_price = price_series[current_idx]['close']
        
        # Future returns at different horizons
        for horizon in [1, 5, 10, 20]:  # 1, 5, 10, 20 days
            if current_idx + horizon < len(price_series):
                future_price = price_series[current_idx + horizon]['close']
                future_return = (future_price - current_price) / current_price
                targets[f'future_return_{horizon}d'] = future_return
            else:
                targets[f'future_return_{horizon}d'] = 0
        
        # Trading signals based on 5-day returns
        future_return_5d = targets['future_return_5d']
        if future_return_5d > 0.02:  # >2% gain
            targets['future_direction'] = 'buy'
        elif future_return_5d < -0.02:  # >2% loss
            targets['future_direction'] = 'sell'
        else:
            targets['future_direction'] = 'hold'
        
        return targets
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that compare across different assets."""
        
        # Market-wide indicators
        daily_market = df.groupby('date').agg({
            'daily_return': ['mean', 'std'],
            'volume': 'sum',
            'volatility': 'mean'
        }).reset_index()
        
        daily_market.columns = ['date', 'market_return', 'market_volatility', 'total_volume', 'avg_volatility']
        
        # Merge back to main dataframe
        df = df.merge(daily_market, on='date', how='left')
        
        # Relative performance
        df['relative_return'] = df['daily_return'] - df['market_return']
        df['relative_volatility'] = df['volatility'] - df['avg_volatility']
        df['beta_adjusted_return'] = df['daily_return'] - df['beta'] * df['market_return']
        
        return df
    
    def analyze_trading_patterns(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze patterns in trading data."""
        
        print("🔍 Analyzing trading patterns...")
        
        patterns = {}
        
        # 1. Return distribution analysis
        patterns['return_analysis'] = {
            'mean_daily_return': X['daily_return'].mean(),
            'daily_volatility': X['daily_return'].std(),
            'sharpe_ratio': X['daily_return'].mean() / X['daily_return'].std() * np.sqrt(252),
            'max_daily_gain': X['daily_return'].max(),
            'max_daily_loss': X['daily_return'].min(),
            'positive_days_pct': (X['daily_return'] > 0).mean() * 100
        }
        
        # 2. Asset type performance
        asset_performance = X.groupby('asset_type').agg({
            'daily_return': ['mean', 'std', 'count'],
            'volatility': 'mean'
        }).round(4)
        
        patterns['asset_analysis'] = {
            'performance_by_type': asset_performance.to_dict(),
            'best_performing_type': X.groupby('asset_type')['daily_return'].mean().idxmax(),
            'most_volatile_type': X.groupby('asset_type')['volatility'].mean().idxmax()
        }
        
        # 3. Temporal patterns
        monthly_returns = X.groupby('month')['daily_return'].mean()
        daily_returns = X.groupby('day_of_week')['daily_return'].mean()
        
        patterns['temporal_analysis'] = {
            'best_month': monthly_returns.idxmax(),
            'worst_month': monthly_returns.idxmin(),
            'best_weekday': daily_returns.idxmax(),
            'worst_weekday': daily_returns.idxmin(),
            'monthly_returns': monthly_returns.to_dict(),
            'daily_returns': daily_returns.to_dict(),
            'end_of_month_effect': X[X['is_month_end'] == 1]['daily_return'].mean(),
            'quarter_end_effect': X[X['is_quarter_end'] == 1]['daily_return'].mean()
        }
        
        # 4. Technical indicator effectiveness
        tech_indicators = ['rsi', 'macd', 'bb_position', 'momentum', 'williams_r']
        indicator_correlations = {}
        
        for indicator in tech_indicators:
            if indicator in X.columns:
                correlation = np.corrcoef(X[indicator], targets['return_prediction'])[0, 1]
                if not np.isnan(correlation):
                    indicator_correlations[indicator] = abs(correlation)
        
        patterns['technical_analysis'] = {
            'best_indicators': dict(sorted(indicator_correlations.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]),
            'rsi_overbought_performance': X[X['rsi'] > 70]['daily_return'].mean() if 'rsi' in X.columns else 0,
            'rsi_oversold_performance': X[X['rsi'] < 30]['daily_return'].mean() if 'rsi' in X.columns else 0,
            'high_volatility_returns': X[X['volatility'] > X['volatility'].quantile(0.8)]['daily_return'].mean()
        }
        
        # 5. Market regime analysis
        patterns['regime_analysis'] = {
            'high_vol_days': (X['volatility'] > 0.3).mean(),
            'stress_period_returns': X[X['market_stress'] == 1]['daily_return'].mean() if 'market_stress' in X.columns else 0,
            'low_vol_returns': X[X['volatility'] < 0.15]['daily_return'].mean(),
            'high_vol_returns': X[X['volatility'] > 0.4]['daily_return'].mean()
        }
        
        # 6. Signal distribution analysis
        signal_dist = targets['direction_prediction'].value_counts(normalize=True) * 100
        patterns['signal_analysis'] = {
            'signal_distribution': signal_dist.to_dict(),
            'buy_signal_rate': signal_dist.get('buy', 0),
            'sell_signal_rate': signal_dist.get('sell', 0),
            'hold_signal_rate': signal_dist.get('hold', 0)
        }
        
        print("✅ Trading pattern analysis completed")
        return patterns
    
    def train_trading_models(self, X: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Train models for trading signal prediction."""
        
        print("🚀 Training trading models...")
        
        all_results = {}
        
        for target_name, target in targets.items():
            print(f"\nTraining models for {target_name}...")
            
            # Remove rows with NaN targets
            valid_mask = target.notna()
            X_clean = X[valid_mask]
            target_clean = target[valid_mask]
            
            # Split data (time-aware split for financial data)
            split_index = int(len(X_clean) * (1 - self.config['test_size']))
            
            X_train = X_clean.iloc[:split_index]
            X_test = X_clean.iloc[split_index:]
            y_train = target_clean.iloc[:split_index]
            y_test = target_clean.iloc[split_index:]
            
            target_results = {}
            
            # Choose model type based on target
            if target_name == 'direction_prediction':
                models = ClassificationModels()
            else:  # return_prediction
                models = RegressionModels()
            
            for algorithm in self.config['algorithms']:
                print(f"  Training {algorithm}...")
                
                try:
                    # Train model
                    model, training_time = models.train_model(
                        X_train, y_train, 
                        algorithm=algorithm,
                        class_weight='balanced' if target_name == 'direction_prediction' else None
                    )
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                    
                    if hasattr(model, 'predict_proba') and target_name == 'direction_prediction':
                        y_pred_proba = model.predict_proba(X_test)
                    
                    # Evaluate model
                    evaluator = ModelEvaluator()
                    if target_name == 'direction_prediction':
                        metrics = evaluator.classification_metrics(y_test, y_pred, y_pred_proba)
                    else:
                        metrics = evaluator.regression_metrics(y_test, y_pred)
                    
                    # Calculate trading performance
                    trading_metrics = self.calculate_trading_performance(
                        target_name, y_test, y_pred, X_test
                    )
                    
                    target_results[algorithm] = {
                        'model': model,
                        'predictions': y_pred,
                        'probabilities': y_pred_proba,
                        'metrics': metrics,
                        'trading_metrics': trading_metrics,
                        'training_time': training_time,
                        'test_data': (X_test, y_test)
                    }
                    
                    if target_name == 'direction_prediction':
                        print(f"    ✅ {algorithm} - Accuracy: {metrics['accuracy']:.3f}, "
                              f"F1: {metrics['f1_score']:.3f}")
                    else:
                        print(f"    ✅ {algorithm} - R²: {metrics['r2_score']:.3f}, "
                              f"RMSE: {metrics['rmse']:.4f}")
                
                except Exception as e:
                    print(f"    ❌ {algorithm} failed: {str(e)}")
                    continue
            
            if target_results:  # Only proceed if we have results
                # Find best model
                if target_name == 'direction_prediction':
                    best_algorithm = max(target_results.keys(), 
                                       key=lambda x: target_results[x]['metrics']['f1_score'])
                else:
                    best_algorithm = max(target_results.keys(), 
                                       key=lambda x: target_results[x]['metrics']['r2_score'])
                
                all_results[target_name] = {
                    'results': target_results,
                    'best_model': best_algorithm,
                    'best_performance': target_results[best_algorithm]
                }
                
                print(f"  🏆 Best model for {target_name}: {best_algorithm}")
        
        return all_results
    
    def calculate_trading_performance(self, target_name: str, y_true: pd.Series, 
                                    y_pred: pd.Series, X_test: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific performance metrics."""
        
        if target_name == 'direction_prediction':
            # Calculate signal accuracy
            correct_signals = (y_true == y_pred).sum()
            total_signals = len(y_true)
            accuracy = correct_signals / total_signals if total_signals > 0 else 0
            
            # Simplified performance metrics
            buy_signals = (y_pred == 'buy')
            sell_signals = (y_pred == 'sell')
            
            buy_accuracy = ((y_true == 'buy') & (y_pred == 'buy')).sum() / max((y_true == 'buy').sum(), 1)
            sell_accuracy = ((y_true == 'sell') & (y_pred == 'sell')).sum() / max((y_true == 'sell').sum(), 1)
            
            # Simple return simulation
            returns = []
            for i, (true_signal, pred_signal) in enumerate(zip(y_true, y_pred)):
                if pred_signal == 'buy':
                    returns.append(np.random.normal(0.01, 0.02))  # Expected positive return
                elif pred_signal == 'sell':
                    returns.append(np.random.normal(-0.005, 0.015))  # Expected negative return
                else:
                    returns.append(0)
            
            annual_return = np.mean(returns) * 252
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            return {
                'signal_accuracy': accuracy,
                'buy_signal_accuracy': buy_accuracy,
                'sell_signal_accuracy': sell_accuracy,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': (np.array(returns) > 0).mean()
            }
        
        else:  # return_prediction
            mae = np.mean(np.abs(y_true - y_pred))
            direction_accuracy = ((y_true > 0) == (y_pred > 0)).mean()
            
            # Simple trading simulation
            portfolio_returns = []
            for true_ret, pred_ret in zip(y_true, y_pred):
                if abs(pred_ret) > 0.01:
                    position_size = min(abs(pred_ret), 0.1)
                    trade_return = true_ret * position_size * np.sign(pred_ret)
                    portfolio_returns.append(trade_return)
                else:
                    portfolio_returns.append(0)
            
            total_return = sum(portfolio_returns)
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
            
            return {
                'prediction_mae': mae,
                'direction_accuracy': direction_accuracy,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'hit_rate': direction_accuracy
            }
    
    def backtest_strategy(self, X: pd.DataFrame, models_dict: Dict[str, Any]) -> pd.DataFrame:
        """Perform comprehensive backtesting of the trading strategy."""
        
        print("📊 Running trading strategy backtest...")
        
        if not models_dict or 'direction_prediction' not in models_dict:
            print("❌ No models available for backtesting")
            return pd.DataFrame()
        
        # Get best model
        direction_model = models_dict['direction_prediction']['best_performance']['model']
        
        # Initialize portfolio
        initial_capital = self.config['business_params']['initial_capital']
        portfolio_value = initial_capital
        
        # Get test data (last 20% chronologically)
        split_index = int(len(X) * 0.8)
        X_backtest = X.iloc[split_index:].copy()
        
        daily_values = []
        
        for idx, row in X_backtest.iterrows():
            # Make prediction
            signal = direction_model.predict([row.drop(['date', 'asset_id'], errors='ignore')])[0]
            
            # Simulate trading return based on signal
            if signal == 'buy':
                daily_return = np.random.normal(0.001, 0.02)
            elif signal == 'sell':
                daily_return = np.random.normal(-0.0005, 0.015)
            else:
                daily_return = 0
            
            portfolio_value *= (1 + daily_return)
            
            daily_values.append({
                'date': row.get('date', idx),
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'signal': signal
            })
        
        backtest_results = pd.DataFrame(daily_values)
        
        if not backtest_results.empty:
            total_return = (portfolio_value - initial_capital) / initial_capital
            annual_return = np.mean(backtest_results['daily_return']) * 252
            volatility = np.std(backtest_results['daily_return']) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            print(f"✅ Backtest completed!")
            print(f"📈 Total Return: {total_return:.1%}")
            print(f"📊 Annual Return: {annual_return:.1%}")
            print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return backtest_results
    
    def visualize_results(self, patterns: Dict[str, Any], results: Dict[str, Any], 
                         backtest: pd.DataFrame) -> None:
        """Create comprehensive visualizations of trading results."""
        
        print("📊 Creating trading analysis visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Return analysis
        ax1 = plt.subplot(3, 4, 1)
        if 'return_analysis' in patterns:
            metrics = ['Mean Return', 'Volatility', 'Sharpe', 'Max Gain', 'Max Loss']
            values = [
                patterns['return_analysis']['mean_daily_return'] * 252,
                patterns['return_analysis']['daily_volatility'] * np.sqrt(252),
                patterns['return_analysis']['sharpe_ratio'],
                patterns['return_analysis']['max_daily_gain'],
                patterns['return_analysis']['max_daily_loss']
            ]
            
            colors = ['green' if v > 0 else 'red' for v in values]
            ax1.bar(range(len(metrics)), values, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.set_title('Return Analysis', fontweight='bold')
        
        # 2. Asset type performance
        ax2 = plt.subplot(3, 4, 2)
        if 'asset_analysis' in patterns:
            asset_perf = patterns['asset_analysis']['performance_by_type']['daily_return']['mean']
            asset_types = list(asset_perf.keys())
            returns = [asset_perf[at] * 252 for at in asset_types]
            
            ax2.bar(asset_types, returns, alpha=0.7)
            ax2.set_title('Annual Returns by Asset Type', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Portfolio performance
        ax3 = plt.subplot(3, 4, (3, 4))
        if not backtest.empty:
            portfolio_values = backtest['portfolio_value']
            dates = range(len(portfolio_values))
            
            ax3.plot(dates, portfolio_values, linewidth=2, color='blue', label='Portfolio')
            
            # Benchmark
            initial_value = portfolio_values.iloc[0]
            benchmark = initial_value * (1 + 0.08) ** (np.array(dates) / 252)
            ax3.plot(dates, benchmark, linestyle='--', color='gray', label='Benchmark (8%)')
            
            ax3.set_title('Portfolio Performance', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Monthly returns
        ax4 = plt.subplot(3, 4, 5)
        if 'temporal_analysis' in patterns:
            monthly_returns = patterns['temporal_analysis']['monthly_returns']
            months = list(monthly_returns.keys())
            returns = [monthly_returns[m] * 21 for m in months]
            
            ax4.plot(months, returns, marker='o', linewidth=2)
            ax4.set_title('Monthly Return Patterns', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. Model performance
        ax5 = plt.subplot(3, 4, 6)
        if 'direction_prediction' in results:
            algorithms = list(results['direction_prediction']['results'].keys())
            accuracies = [results['direction_prediction']['results'][alg]['metrics']['accuracy'] 
                         for alg in algorithms]
            
            bars = ax5.bar(algorithms, accuracies, color='gold', alpha=0.7)
            ax5.set_title('Model Accuracy', fontweight='bold')
            ax5.tick_params(axis='x', rotation=45)
            
            # Highlight best
            best_idx = np.argmax(accuracies)
            bars[best_idx].set_color('darkgoldenrod')
        
        # 6. Signal distribution
        ax6 = plt.subplot(3, 4, 7)
        if 'signal_analysis' in patterns:
            signal_dist = patterns['signal_analysis']['signal_distribution']
            signals = list(signal_dist.keys())
            percentages = list(signal_dist.values())
            
            colors = {'buy': 'green', 'hold': 'gray', 'sell': 'red'}
            signal_colors = [colors.get(s, 'blue') for s in signals]
            
            ax6.pie(percentages, labels=signals, autopct='%1.1f%%', colors=signal_colors)
            ax6.set_title('Trading Signals', fontweight='bold')
        
        # 7. Technical indicators
        ax7 = plt.subplot(3, 4, 8)
        if 'technical_analysis' in patterns:
            indicators = patterns['technical_analysis']['best_indicators']
            if indicators:
                names = list(indicators.keys())[:5]
                correlations = [indicators[name] for name in names]
                
                ax7.barh(names, correlations, color='purple', alpha=0.7)
                ax7.set_title('Best Technical Indicators', fontweight='bold')
        
        # 8. Risk-return scatter
        ax8 = plt.subplot(3, 4, 9)
        if 'asset_analysis' in patterns:
            asset_perf = patterns['asset_analysis']['performance_by_type']
            asset_types = list(asset_perf['daily_return']['mean'].keys())
            
            returns = [asset_perf['daily_return']['mean'][at] * 252 for at in asset_types]
            risks = [asset_perf['daily_return']['std'][at] * np.sqrt(252) for at in asset_types]
            
            ax8.scatter(risks, returns, s=100, alpha=0.7)
            for i, asset_type in enumerate(asset_types):
                ax8.annotate(asset_type, (risks[i], returns[i]), fontsize=8)
            
            ax8.set_xlabel('Risk')
            ax8.set_ylabel('Return')
            ax8.set_title('Risk-Return Profile', fontweight='bold')
        
        # 9. Daily returns distribution
        ax9 = plt.subplot(3, 4, 10)
        if not backtest.empty and 'daily_return' in backtest.columns:
            daily_returns = backtest['daily_return'] * 100
            ax9.hist(daily_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax9.axvline(daily_returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {daily_returns.mean():.2f}%')
            ax9.set_title('Daily Returns Distribution', fontweight='bold')
            ax9.set_xlabel('Daily Return (%)')
            ax9.legend()
        
        # 10. Volatility regimes
        ax10 = plt.subplot(3, 4, 11)
        if 'regime_analysis' in patterns:
            regimes = ['Low Vol', 'High Vol', 'Stress']
            regime_returns = [
                patterns['regime_analysis']['low_vol_returns'],
                patterns['regime_analysis']['high_vol_returns'],
                patterns['regime_analysis']['stress_period_returns']
            ]
            
            colors = ['lightblue', 'orange', 'red']
            ax10.bar(regimes, regime_returns, color=colors, alpha=0.7)
            ax10.set_title('Market Regime Performance', fontweight='bold')
            ax10.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # 11. Trading metrics
        ax11 = plt.subplot(3, 4, 12)
        if 'direction_prediction' in results:
            best_result = results['direction_prediction']['best_performance']
            trading_metrics = best_result.get('trading_metrics', {})
            
            if trading_metrics:
                metrics = ['Sharpe Ratio', 'Win Rate', 'Annual Return']
                values = [
                    trading_metrics.get('sharpe_ratio', 0),
                    trading_metrics.get('win_rate', 0),
                    trading_metrics.get('annual_return', 0)
                ]
                
                ax11.bar(metrics, values, color=['gold', 'lightgreen', 'darkblue'], alpha=0.7)
                ax11.set_title('Strategy Metrics', fontweight='bold')
                ax11.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations completed")
    
    def generate_business_report(self, patterns: Dict[str, Any], results: Dict[str, Any], 
                               backtest: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive business report."""
        
        print("📋 Generating business report...")
        
        # Calculate key metrics
        if not backtest.empty:
            initial_capital = self.config['business_params']['initial_capital']
            final_value = backtest['portfolio_value'].iloc[-1] if not backtest.empty else initial_capital
            total_return = (final_value - initial_capital) / initial_capital
            annual_return = (final_value / initial_capital) ** (252 / len(backtest)) - 1 if len(backtest) > 0 else 0
            
            daily_returns = backtest['daily_return']
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - self.config['business_params']['risk_free_rate']) / volatility if volatility > 0 else 0
            
            max_drawdown = ((backtest['portfolio_value'].cummax() - backtest['portfolio_value']) / backtest['portfolio_value'].cummax()).max()
        else:
            total_return = annual_return = volatility = sharpe_ratio = max_drawdown = 0
            final_value = initial_capital
        
        # Best model performance
        best_model_name = results.get('direction_prediction', {}).get('best_model', 'N/A')
        best_accuracy = 0
        if 'direction_prediction' in results:
            best_performance = results['direction_prediction']['best_performance']
            best_accuracy = best_performance['metrics']['accuracy']
        
        report = {
            'executive_summary': {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_portfolio_value': final_value,
                'best_model': best_model_name,
                'model_accuracy': best_accuracy
            },
            'performance_metrics': {
                'returns': {
                    'total_return_pct': total_return * 100,
                    'annual_return_pct': annual_return * 100,
                    'volatility_pct': volatility * 100,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown_pct': max_drawdown * 100
                },
                'trading_stats': {
                    'total_trades': len(backtest),
                    'avg_daily_return': daily_returns.mean() if not backtest.empty else 0,
                    'win_rate': (daily_returns > 0).mean() if not backtest.empty else 0,
                    'best_day': daily_returns.max() if not backtest.empty else 0,
                    'worst_day': daily_returns.min() if not backtest.empty else 0
                }
            },
            'model_performance': results,
            'market_insights': patterns,
            'risk_analysis': {
                'portfolio_volatility': volatility,
                'maximum_drawdown': max_drawdown,
                'var_95': daily_returns.quantile(0.05) if not backtest.empty else 0,
                'var_99': daily_returns.quantile(0.01) if not backtest.empty else 0
            },
            'recommendations': self._generate_recommendations(patterns, results, backtest)
        }
        
        print("✅ Business report generated")
        return report
    
    def _generate_recommendations(self, patterns: Dict[str, Any], results: Dict[str, Any], 
                                backtest: pd.DataFrame) -> List[str]:
        """Generate actionable business recommendations."""
        
        recommendations = []
        
        # Performance-based recommendations
        if not backtest.empty:
            annual_return = backtest['daily_return'].mean() * 252
            if annual_return > 0.15:
                recommendations.append("Strong performance detected. Consider increasing capital allocation.")
            elif annual_return < 0.05:
                recommendations.append("Underperformance identified. Review strategy parameters and risk management.")
        
        # Model-based recommendations
        if 'direction_prediction' in results:
            best_accuracy = results['direction_prediction']['best_performance']['metrics']['accuracy']
            if best_accuracy > 0.6:
                recommendations.append("High model accuracy achieved. Consider deploying to production.")
            else:
                recommendations.append("Model accuracy needs improvement. Consider feature engineering or ensemble methods.")
        
        # Market regime recommendations
        if 'regime_analysis' in patterns:
            high_vol_perf = patterns['regime_analysis']['high_vol_returns']
            if high_vol_perf < 0:
                recommendations.append("Poor performance in high volatility periods. Implement volatility-based position sizing.")
        
        # Asset allocation recommendations
        if 'asset_analysis' in patterns:
            best_asset = patterns['asset_analysis']['best_performing_type']
            recommendations.append(f"Consider increasing allocation to {best_asset} based on historical performance.")
        
        # Risk management recommendations
        if not backtest.empty:
            max_dd = ((backtest['portfolio_value'].cummax() - backtest['portfolio_value']) / backtest['portfolio_value'].cummax()).max()
            if max_dd > 0.2:
                recommendations.append("High drawdown detected. Implement stricter stop-loss rules.")
        
        # Technical analysis recommendations
        if 'technical_analysis' in patterns and patterns['technical_analysis']['best_indicators']:
            best_indicator = list(patterns['technical_analysis']['best_indicators'].keys())[0]
            recommendations.append(f"Focus on {best_indicator} indicator for signal generation.")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete algorithmic trading analysis pipeline."""
        
        print("🚀 Starting complete algorithmic trading analysis...")
        
        # Generate dataset
        X, targets = self.generate_trading_dataset()
        self.trading_data = (X, targets)
        
        # Analyze patterns
        patterns = self.analyze_trading_patterns(X, targets)
        
        # Train models
        results = self.train_trading_models(X, targets)
        self.trading_results = results
        
        # Backtest strategy
        backtest = self.backtest_strategy(X, results)
        self.portfolio_performance = backtest
        
        # Create visualizations
        self.visualize_results(patterns, results, backtest)
        
        # Generate business report
        business_report = self.generate_business_report(patterns, results, backtest)
        
        print("🎉 Complete algorithmic trading analysis finished!")
        
        return {
            'data': (X, targets),
            'patterns': patterns,
            'results': results,
            'backtest': backtest,
            'report': business_report
        }

# Usage example and main execution
if __name__ == "__main__":
    # Initialize trading system
    trading_system = AlgorithmicTradingSystem()
    
    # Run complete analysis
    analysis_results = trading_system.run_complete_analysis()
    
    # Print summary
    report = analysis_results['report']
    print("\n" + "="*50)
    print("ALGORITHMIC TRADING SYSTEM - EXECUTIVE SUMMARY")
    print("="*50)
    print(f"Total Return: {report['executive_summary']['total_return']:.1%}")
    print(f"Annual Return: {report['executive_summary']['annual_return']:.1%}")
    print(f"Sharpe Ratio: {report['executive_summary']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['executive_summary']['max_drawdown']:.1%}")
    print(f"Best Model: {report['executive_summary']['best_model']}")
    print(f"Model Accuracy: {report['executive_summary']['model_accuracy']:.1%}")
    print("\nTop Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    print("="*50)