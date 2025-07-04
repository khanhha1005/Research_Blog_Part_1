# Arbitrage Prediction Model 

This guide explains how to save the trained model and use it for making predictions on new data.

## Files Overview

- `arbitrage_predictor_v3.py` - Training script (enhanced with model saving)
- `inference.py` - Inference script for making predictions
- `example_inference.py` - Example usage demonstration
- `INFERENCE_README.md` - This guide

## Step 1: Train and Save the Model

First, run the training script to train the model and save it:

```bash
python arbitrage_predictor_v3.py
```

This will:
1. Train the LSTM model on your data
2. Save the best model with all necessary metadata to `arbitrage_profit_model_v3.pth`
3. The saved model includes:
   - Model weights and architecture
   - Feature column names
   - Sequence length configuration
   - Model hyperparameters

## Step 2: Make Predictions

### Option A: Command Line Interface

Use the inference script with command line arguments:

```bash
# Batch prediction from CSV file
python inference.py --data your_new_data.csv --output predictions.csv

# Live prediction (single sample)
python inference.py --data sample.csv --mode live
```

### Option B: Python API

Use the inference class directly in your code:

```python
from inference import ArbitrageInference

# Load the trained model
inference = ArbitrageInference('arbitrage_profit_model_v3.pth')

# Make batch predictions from CSV
results = inference.predict_from_csv('new_data.csv', 'output.csv')

# Make live prediction on single sample
live_data = {
    'price_HYPE_HyperEVM': 1.25,
    'price_HYPE_HyperCORE': 1.30,
    'Amount_HYPE_HyperEVM': 1000.0,
    'delta_USD': 0.05
}
prediction = inference.predict_live_data(live_data)
print(f"Predicted profit: ${prediction:.4f}")
```

### Option C: Run the Example

Run the example script to see everything in action:

```bash
python example_inference.py
```

This will:
1. Load the trained model
2. Create sample data
3. Make predictions
4. Show results and save files

## Input Data Format

Your input CSV should contain these columns:
- `datetime` (optional, for timestamps)
- `price_HYPE_HyperEVM`
- `price_HYPE_HyperCORE`
- `Amount_HYPE_HyperEVM`
- `delta_USD`

The inference script will automatically create the derived features used during training.

## Output Format

The predictions are saved as a CSV with:
- `timestamp` - Time of prediction
- `predicted_arbitrage_profit` - Predicted arbitrage profit in USD
- `original_*` - Original input values (if available)

## Model Requirements

- **Sequence Length**: 200 data points (minimum)
- **Features**: 10 features (4 base + 6 derived)
- **Output**: Single value (arbitrage profit prediction)

## Error Handling

The inference script includes robust error handling:
- Checks for minimum data requirements
- Handles missing or NaN values
- Provides informative error messages
- Graceful fallbacks for edge cases

## Performance Notes

- The model requires at least 200 data points for each prediction
- Predictions are made sequentially using sliding windows
- GPU acceleration is used if available
- Memory usage scales with sequence length and batch size

## Troubleshooting

### Common Issues:

1. **Model file not found**
   - Ensure you've run the training script first
   - Check the model file path

2. **Insufficient data**
   - Need at least 200 data points for predictions
   - Add more historical data if needed

3. **Missing columns**
   - Ensure your CSV has the required columns
   - Check column names match exactly

4. **Memory issues**
   - Reduce batch size if processing large datasets
   - Use CPU if GPU memory is insufficient

## Example Usage Scenarios

### Scenario 1: Real-time Trading
```python
# Load model once
inference = ArbitrageInference('model.pth')

# Make predictions on live data
while True:
    live_data = get_latest_market_data()
    profit = inference.predict_live_data(live_data)
    if profit > threshold:
        execute_trade()
```

### Scenario 2: Batch Analysis
```python
# Analyze historical data
results = inference.predict_from_csv('historical_data.csv', 'analysis.csv')
print(f"Average predicted profit: ${results['predicted_arbitrage_profit'].mean():.2f}")
```

### Scenario 3: Model Comparison
```python
# Compare different models
models = ['model_v1.pth', 'model_v2.pth', 'model_v3.pth']
for model in models:
    inference = ArbitrageInference(model)
    results = inference.predict_from_csv('test_data.csv')
    print(f"{model}: {results['predicted_arbitrage_profit'].mean():.4f}")
```

## Advanced Usage

### Custom Feature Engineering
You can modify the `create_features` method in the inference class to add custom features:

```python
def create_features(self, df):
    df = df.copy()
    # Add your custom features here
    df['custom_feature'] = df['price_HYPE_HyperEVM'] * df['Amount_HYPE_HyperEVM']
    return df
```

### Batch Processing
For large datasets, you can process in chunks:

```python
chunk_size = 1000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    results = inference.predict_batch(chunk)
    # Process results
```

This setup provides a complete pipeline from training to inference, making it easy to deploy your arbitrage prediction model in production environments. 