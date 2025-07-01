#!/usr/bin/env python3
"""
Simple Arbitrage Profit Regression Model
========================================

Predicts "Expected Arbitrage Profit" = Probability of Arbitrage × Potential Volume
Ranks tokens by arbitrage opportunity for next 24 hours.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class ArbitrageRegressor:
    def __init__(self):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(n_estimators=100, random_state=42)
        # self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
        
    def load_and_prepare_data(self, csv_path="final_data_task1_swell.csv"):
        """Load data and create target variable"""
        print("📊 Loading data...")
        df = pd.read_csv(csv_path)
        
        # Create features
        df['price_diff_pct'] = (df['delta_USD'] / df['price_HYPE_HyperCORE']) * 100
        df['volume_usd'] = df['Amount_HYPE_HyperEVM'] * df['price_HYPE_HyperEVM']
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        
        # Target: Expected Arbitrage Profit (actual profit as proxy)
        # In real scenario, this would be: probability × potential_volume
        df['expected_profit'] = df['arb_profit']
        
        # Features for prediction
        features = ['price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'delta_USD', 
                   'price_diff_pct', 'Amount_HYPE_HyperEVM', 'volume_usd', 'hour']
        
        X = df[features].fillna(0)
        y = df['expected_profit']
        
        print(f"   Data shape: {df.shape}")
        print(f"   Features: {features}")
        print(f"   Target range: ${y.min():.2f} to ${y.max():.2f}")
        
        return X, y, df
    
    def train(self, X, y):
        """Train regression model"""
        print("🚀 Training model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        print(f"   Train R²: {r2_score(y_train, train_pred):.4f}")
        print(f"   Test R²:  {r2_score(y_test, test_pred):.4f}")
        print(f"   Test RMSE: ${np.sqrt(mean_squared_error(y_test, test_pred)):.2f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Feature Importance:")
        print(importance.to_string(index=False))
        
        self.is_trained = True
        return importance
    
    def predict_arbitrage_opportunities(self, X):
        """Predict expected arbitrage profit"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.predict(X)
    
    def save_model(self, filepath="arbitrage_regressor.pkl"):
        """Save model"""
        joblib.dump(self.model, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath="arbitrage_regressor.pkl"):
        """Load model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"📥 Model loaded from {filepath}")

def rank_arbitrage_opportunities():
    """Main function to train model and rank opportunities"""
    print("=" * 50)
    print("🎯 Arbitrage Opportunity Ranking System")
    print("=" * 50)
    
    # Initialize and train
    regressor = ArbitrageRegressor()
    X, y, df = regressor.load_and_prepare_data()
    importance = regressor.train(X, y)
    regressor.save_model()
    
    # Predict on recent data (simulate next 24h opportunities)
    print("\n🔮 Top Arbitrage Opportunities (Recent Data):")
    recent_data = df.tail(50)  # Last 50 records as example
    recent_X = recent_data[X.columns].fillna(0)
    predictions = regressor.predict_arbitrage_opportunities(recent_X)
    
    # Create ranking
    ranking_df = pd.DataFrame({
        'price_EVM': recent_data['price_HYPE_HyperEVM'].values,
        'price_Core': recent_data['price_HYPE_HyperCORE'].values,
        'volume_HYPE': recent_data['Amount_HYPE_HyperEVM'].values,
        'actual_profit': recent_data['arb_profit'].values,
        'predicted_profit': predictions,
        'hour': recent_data['hour'].values
    }).sort_values('predicted_profit', ascending=False)
    
    # Clean display for ranking
    display_ranking = ranking_df.head(10).copy()
    display_ranking.columns = ['Price_EVM', 'Price_Core', 'Volume_HYPE', 'Actual_Profit', 'Predicted_Profit', 'Hour']
    
    print("\nTop 10 Opportunities:")
    print(display_ranking.round(2))
    
    # Summary statistics
    print(f"\n📊 Summary:")
    print(f"   Avg predicted profit: ${predictions.mean():.2f}")
    print(f"   Max predicted profit: ${predictions.max():.2f}")
    print(f"   Opportunities > $10: {(predictions > 10).sum()}")
    print(f"   Opportunities > $100: {(predictions > 100).sum()}")
    
    return regressor, ranking_df

def predict_new_scenario():
    """Example: Predict for hypothetical scenarios"""
    print("\n" + "=" * 40)
    print("🚀 Predict New Scenarios")
    print("=" * 40)
    
    # Load trained model
    regressor = ArbitrageRegressor()
    try:
        regressor.load_model()
    except:
        print("❌ No trained model found. Run main function first.")
        return
    
    # Example scenarios for next 24h
    scenarios = pd.DataFrame({
        'price_HYPE_HyperEVM': [35.0, 34.5, 36.0, 33.8],
        'price_HYPE_HyperCORE': [34.8, 34.9, 35.7, 34.0],
        'delta_USD': [0.2, -0.4, 0.3, -0.2],
        'price_diff_pct': [0.57, -1.15, 0.84, -0.59],
        'Amount_HYPE_HyperEVM': [100, 500, 1000, 250],
        'volume_usd': [3500, 17250, 36000, 8450],
        'hour': [9, 14, 18, 22]
    })
    
    predictions = regressor.predict_arbitrage_opportunities(scenarios)
    
    scenarios['expected_profit'] = predictions
    scenarios_ranked = scenarios.sort_values('expected_profit', ascending=False)
    
    # Clean display
    display_df = scenarios_ranked[['price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'Amount_HYPE_HyperEVM', 'expected_profit']].copy()
    display_df.columns = ['Price_EVM', 'Price_Core', 'Volume_HYPE', 'Expected_Profit']
    
    print("Predicted Opportunities (Ranked):")
    print(display_df.round(2))

if __name__ == "__main__":
    # Train model and rank opportunities
    regressor, ranking_df = rank_arbitrage_opportunities()
    
    # Show prediction examples
    predict_new_scenario() 