import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMArbitrageModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, dropout=0.2):
        super(LSTMArbitrageModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.tanh1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        
        return out

class ArbitrageInference:
    def __init__(self, model_path):
        """Initialize inference with a trained model"""
        self.model = None
        self.feature_columns = []
        self.sequence_length = 200
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model from file"""
        if not torch.cuda.is_available():
            model_data = torch.load(model_path, map_location='cpu')
        else:
            model_data = torch.load(model_path)
        
        # Load configurations
        model_config = model_data['model_config']
        predictor_config = model_data['predictor_config']
        
        # Update attributes
        self.sequence_length = predictor_config['sequence_length']
        self.feature_columns = predictor_config['feature_columns']
        
        # Initialize and load model
        self.model = LSTMArbitrageModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout=model_config['dropout']
        ).to(device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Model config: {model_config}")
        print(f"Feature columns: {self.feature_columns}")
        print(f"Sequence length: {self.sequence_length}")
    
    def create_features(self, df):
        """Create the same features used during training"""
        df = df.copy()
        df['price_ratio'] = df['price_HYPE_HyperCORE'] / df['price_HYPE_HyperEVM']
        df['price_spread'] = df['price_HYPE_HyperCORE'] - df['price_HYPE_HyperEVM']
        df['ma_ratio_3'] = df['price_ratio'].rolling(window=3).mean()
        df['ma_delta_3'] = df['delta_USD'].rolling(window=3).mean()
        df['price_change'] = df['price_HYPE_HyperEVM'].pct_change()
        df['ratio_change'] = df['price_ratio'].pct_change()
        df = df.fillna(method='ffill').fillna(0)
        return df
    
    def predict_single(self, window_data):
        """Make a single prediction on a window of data"""
        if len(window_data) < self.sequence_length:
            print(f"Warning: Need at least {self.sequence_length} data points, got {len(window_data)}")
            return 0.0
            
        features = window_data[self.feature_columns].values
        
        # Handle NaN values
        features = np.nan_to_num(features)
        
        # Get the last sequence_length data points
        sequence = features[-self.sequence_length:]
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            prediction = self.model(X)
            
            if torch.isnan(prediction).any():
                return 0.0
        
        # Get prediction value
        pred_value = prediction.cpu().numpy()[0][0]
        
        return max(0.0, pred_value) if not np.isnan(pred_value) else 0.0
    
    def predict_batch(self, df, start_idx=None, end_idx=None):
        """Make predictions on a batch of data"""
        if start_idx is None:
            start_idx = self.sequence_length
        if end_idx is None:
            end_idx = len(df)
        
        predictions = []
        timestamps = []
        
        for i in range(start_idx, end_idx):
            try:
                window_data = df.iloc[i-self.sequence_length:i]
                pred = self.predict_single(window_data)
                timestamp = df.iloc[i]['datetime'] if 'datetime' in df.columns else i
                
                predictions.append(pred)
                timestamps.append(timestamp)
            except Exception as e:
                print(f"Error predicting at index {i}: {e}")
                predictions.append(0.0)
                timestamps.append(i)
        
        return predictions, timestamps
    
    def predict_from_csv(self, csv_path, output_path=None):
        """Load data from CSV and make predictions"""
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create features
        df = self.create_features(df)
        
        # Make predictions
        predictions, timestamps = self.predict_batch(df)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_arbitrage_profit': predictions
        })
        
        # Add original data if available
        if len(df) >= len(predictions) + self.sequence_length:
            original_data = df.iloc[self.sequence_length:self.sequence_length+len(predictions)]
            for col in ['price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'Amount_HYPE_HyperEVM', 'delta_USD']:
                if col in original_data.columns:
                    results_df[f'original_{col}'] = original_data[col].values
        
        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return results_df
    
    def predict_live_data(self, data_dict):
        """Make prediction on live data (single sample)"""
        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Create features
        df = self.create_features(df)
        
        # For single sample, we need to pad with zeros or use a different approach
        if len(df) < self.sequence_length:
            # Pad with zeros (you might want to use a different strategy)
            padding = pd.DataFrame(0, index=range(self.sequence_length - len(df)), columns=df.columns)
            df = pd.concat([padding, df], ignore_index=True)
        
        # Make prediction
        prediction = self.predict_single(df)
        
        return prediction

def main():
    """Example usage of the inference script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Arbitrage Profit Prediction Inference')
    parser.add_argument('--model', type=str, default='arbitrage_profit_model_v3.pth',
                       help='Path to the trained model file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the CSV file with new data')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--mode', type=str, choices=['batch', 'live'], default='batch',
                       help='Inference mode: batch (CSV file) or live (single sample)')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inference = ArbitrageInference(args.model)
        
        if args.mode == 'batch':
            # Batch prediction from CSV
            results = inference.predict_from_csv(args.data, args.output)
            
            print("\nPrediction Summary:")
            print(f"Total predictions: {len(results)}")
            print(f"Average predicted profit: ${results['predicted_arbitrage_profit'].mean():.4f}")
            print(f"Max predicted profit: ${results['predicted_arbitrage_profit'].max():.4f}")
            print(f"Min predicted profit: ${results['predicted_arbitrage_profit'].min():.4f}")
            
            # Show first few predictions
            print("\nFirst 10 predictions:")
            print(results.head(10))
            
        elif args.mode == 'live':
            # Live prediction (example with sample data)
            sample_data = {
                'price_HYPE_HyperEVM': 1.25,
                'price_HYPE_HyperCORE': 1.30,
                'Amount_HYPE_HyperEVM': 1000.0,
                'delta_USD': 0.05
            }
            
            prediction = inference.predict_live_data(sample_data)
            print(f"Live prediction: ${prediction:.4f}")
    
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found. Please train the model first.")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main() 