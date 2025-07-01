# Research_Blog_Part1 : Arbitrage Opportunity Prediction System

**Simple regression model to predict arbitrage profits and rank opportunities.**

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r regression_requirements.txt
   ```

2. **Run the model**:
   ```bash
   python arbitrage_regression.py
   ```

## What It Does

- **Loads your CSV data** (`final_data_task1_swell.csv`)
- **Creates 7 key features** from price/volume/time data
- **Trains a Random Forest** regression model
- **Predicts expected arbitrage profit** (in USD)
- **Ranks opportunities** by profit potential

## Key Features Used

- `price_HYPE_HyperEVM` - EVM chain price
- `price_HYPE_HyperCORE` - Core chain price  
- `delta_USD` - Price difference in USD
- `price_diff_pct` - Price difference percentage
- `Amount_HYPE_HyperEVM` - Trade volume
- `volume_usd` - USD volume
- `hour` - Hour of day

## Output

```
🔮 Top Arbitrage Opportunities:
   Price_EVM  Price_Core  Volume_HYPE  Predicted_Profit
1      16.85       17.02         94.6             15.21
2      16.86       17.02        148.9             24.10
3      16.71       16.80        720.0             44.00

📊 Summary:
   Avg predicted profit: $5.23
   Max predicted profit: $140.14
   Opportunities > $10: 45
   Opportunities > $100: 3
```

## Customization

**Change model**: Replace `RandomForestRegressor` with `XGBRegressor` or others

**Add features**: Modify the `features` list in `load_and_prepare_data()`

**Adjust threshold**: Change profit thresholds in summary statistics

## Files

- `arbitrage_regression.py` - Main model code
- `regression_requirements.txt` - Dependencies
- `arbitrage_regressor.pkl` - Saved model (created after first run)

**Target**: Ranks tokens by **Expected Arbitrage Profit = Probability × Volume** 