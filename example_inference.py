#!/usr/bin/env python3
"""
Example script demonstrating how to use the arbitrage prediction inference
"""

import pandas as pd
import numpy as np
from inference import ArbitrageInference

def create_sample_data(n_samples=300):
    """Create sample data for testing inference"""
    np.random.seed(42)
    
    # Generate realistic sample data
    base_price_evm = 1.25
    base_price_core = 1.30
    
    data = []
    for i in range(n_samples):
        # Simulate price movements
        price_evm = base_price_evm + np.random.normal(0, 0.02)
        price_core = base_price_core + np.random.normal(0, 0.02)
        
        # Ensure prices are positive
        price_evm = max(0.1, price_evm)
        price_core = max(0.1, price_core)
        
        # Simulate volume and delta
        amount_evm = np.random.uniform(500, 2000)
        delta_usd = np.random.normal(0.05, 0.1)
        
        # Calculate arbitrage profit (simplified)
        arb_profit = max(0, (price_core - price_evm) * amount_evm - abs(delta_usd))
        
        data.append({
            'datetime': pd.Timestamp.now() - pd.Timedelta(minutes=n_samples-i),
            'price_HYPE_HyperEVM': price_evm,
            'price_HYPE_HyperCORE': price_core,
            'Amount_HYPE_HyperEVM': amount_evm,
            'delta_USD': delta_usd,
            'arb_profit': arb_profit
        })
    
    return pd.DataFrame(data)

def main():
    """Main example function"""
    print("=== Arbitrage Prediction Inference Example ===\n")
    
    # Check if model exists
    model_path = 'arbitrage_profit_model_v3.pth'
    
    try:
        # Initialize inference
        print("Loading trained model...")
        inference = ArbitrageInference(model_path)
        print("✓ Model loaded successfully!\n")
        
        # Create sample data
        print("Creating sample data...")
        sample_df = create_sample_data(300)
        sample_df.to_csv('sample_data.csv', index=False)
        print("✓ Sample data created and saved to 'sample_data.csv'\n")
        
        # Make batch predictions
        print("Making batch predictions...")
        results = inference.predict_from_csv('sample_data.csv', 'sample_predictions.csv')
        
        print("\n=== Prediction Results ===")
        print(f"Total predictions: {len(results)}")
        print(f"Average predicted profit: ${results['predicted_arbitrage_profit'].mean():.4f}")
        print(f"Max predicted profit: ${results['predicted_arbitrage_profit'].max():.4f}")
        print(f"Min predicted profit: ${results['predicted_arbitrage_profit'].min():.4f}")
        
        # Show first few predictions
        print("\nFirst 10 predictions:")
        print(results.head(10).to_string(index=False))
        
        # Live prediction example
        print("\n=== Live Prediction Example ===")
        live_data = {
            'price_HYPE_HyperEVM': 1.28,
            'price_HYPE_HyperCORE': 1.35,
            'Amount_HYPE_HyperEVM': 1500.0,
            'delta_USD': 0.08
        }
        
        live_prediction = inference.predict_live_data(live_data)
        print(f"Live data: {live_data}")
        print(f"Predicted arbitrage profit: ${live_prediction:.4f}")
        
        print("\n=== Files Created ===")
        print("✓ sample_data.csv - Sample input data")
        print("✓ sample_predictions.csv - Prediction results")
        print("✓ Model file: arbitrage_profit_model_v3.pth")
        
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please run the training script first:")
        print("python arbitrage_predictor_v3.py")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main() 