import pandas as pd
import numpy as np

def simple_arbitrage_prediction(csv_file):
    """
    Đơn giản hóa việc predict arbitrage cho demo - không cần PyTorch
    """
    # Load dữ liệu
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Loaded {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Tạo simple features
    df['price_ratio'] = df['price_HYPE_HyperCORE'] / df['price_HYPE_HyperEVM']
    df['arb_profit_pct'] = (df['arb_profit'] / df['Amount_HYPE_HyperEVM']) * 100
    df['delta_abs'] = abs(df['delta_USD'])
    
    # Simple scoring dựa trên historical patterns
    df['volatility_score'] = df['delta_USD'].rolling(window=5).std().fillna(0)
    df['profit_momentum'] = df['arb_profit'].rolling(window=3).mean().fillna(0)
    df['price_stability'] = 1 / (df['volatility_score'] + 0.001)
    
    # Arbitrage Score = Profit Potential * Price Stability
    df['arbitrage_score'] = df['profit_momentum'] * df['price_stability']
    
    # Normalize score to 0-100
    if df['arbitrage_score'].max() > 0:
        df['arbitrage_score'] = (df['arbitrage_score'] / df['arbitrage_score'].max()) * 100
    
    # Ranking top opportunities
    top_opportunities = df.nlargest(20, 'arbitrage_score')[
        ['datetime', 'arb_profit', 'delta_USD', 'price_ratio', 'arbitrage_score']
    ].round(4)
    
    print("\n=== TOP 20 ARBITRAGE OPPORTUNITIES ===")
    print(top_opportunities.to_string(index=False))
    
    # Future prediction based on trends
    recent_data = df.tail(10)
    avg_profit = recent_data['arb_profit'].mean()
    trend_profit = recent_data['arb_profit'].iloc[-1] - recent_data['arb_profit'].iloc[0]
    
    predicted_profits = []
    for i in range(6):  # Predict next 6 periods
        pred = avg_profit + (trend_profit * (i + 1) * 0.1)  # Simple trend extrapolation
        predicted_profits.append(max(0, pred))  # Không cho phép profit âm
    
    print(f"\n=== PREDICTION FOR NEXT 6 PERIODS ===")
    for i, pred in enumerate(predicted_profits):
        print(f"Period {i+1}: ${pred:.4f}")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total records analyzed: {len(df)}")
    print(f"Average arbitrage profit: ${df['arb_profit'].mean():.4f}")
    print(f"Max arbitrage profit: ${df['arb_profit'].max():.4f}")
    print(f"Best opportunity score: {df['arbitrage_score'].max():.2f}/100")
    print(f"Total historical profit: ${df['arb_profit'].sum():.2f}")
    
    # Token ranking (vì chỉ có HYPE nên ranking theo time periods)
    print(f"\n=== TOKEN ANALYSIS ===")
    print(f"Token: HYPE")
    print(f"- Arbitrage opportunities: {len(df[df['arb_profit'] > 0])}")
    print(f"- Success rate: {(len(df[df['arb_profit'] > 0]) / len(df) * 100):.1f}%")
    print(f"- Average profit per opportunity: ${df[df['arb_profit'] > 0]['arb_profit'].mean():.4f}")
    
    return df, top_opportunities, predicted_profits

def main():
    print("=== SIMPLE ARBITRAGE PREDICTOR ===")
    
    # Run prediction
    try:
        df, top_opps, predictions = simple_arbitrage_prediction('final_data_task1_swell.csv')
        
        # Save results
        top_opps.to_csv('top_arbitrage_opportunities.csv', index=False)
        print(f"\nResults saved to 'top_arbitrage_opportunities.csv'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the CSV file exists and has the correct format")

if __name__ == "__main__":
    main()