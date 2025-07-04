from networkx import tree_data
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArbitrageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

class ArbitragePredictor:
    def __init__(self, sequence_length=200):
        self.sequence_length = sequence_length
        self.model = None
        self.feature_columns = []
        
    def create_features(self, df):
        df = df.copy()
        df['price_ratio'] = df['price_HYPE_HyperCORE'] / df['price_HYPE_HyperEVM']
        df['price_spread'] = df['price_HYPE_HyperCORE'] - df['price_HYPE_HyperEVM']
        df['ma_ratio_3'] = df['price_ratio'].rolling(window=3).mean()
        df['ma_delta_3'] = df['delta_USD'].rolling(window=3).mean()
        df['price_change'] = df['price_HYPE_HyperEVM'].pct_change()
        df['ratio_change'] = df['price_ratio'].pct_change()
        df = df.fillna(method='ffill').fillna(0)
        return df
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df = self.create_features(df)
        
        base_features = ['price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'Amount_HYPE_HyperEVM', 'delta_USD']
        derived_features = ['price_ratio', 'price_spread', 'ma_ratio_3', 'ma_delta_3', 'price_change', 'ratio_change']
        self.feature_columns = base_features + derived_features
        
        return df
    
    def create_sequences(self, data, target_col='arb_profit'):
        features = data[self.feature_columns].values
        target = data[target_col].values
        
        # No clipping or scaling - use raw data directly
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, df, validation_split=0.2, epochs=50, batch_size=16, lr=0.001):
        X, y = self.create_sequences(df)
        
        # Handle NaN
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Target range: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = ArbitrageDataset(X_train, y_train)
        val_dataset = ArbitrageDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMArbitrageModel(input_size, hidden_size=256, num_layers=5, 
                                       output_size=1, dropout=0.2).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        print("Training...")
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    output = self.model(batch_X)
                    loss = criterion(output.squeeze(), batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            scheduler.step(val_loss)
            
            # Print every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    def save_model(self, filepath):
        """Save the trained model with all necessary metadata for inference"""
        if self.model is None:
            raise ValueError("No trained model to save. Please train the model first.")
        
        # Prepare model data
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': 1,
                'dropout': 0.2
            },
            'predictor_config': {
                'sequence_length': self.sequence_length,
                'feature_columns': self.feature_columns
            }
        }
        
        torch.save(model_data, filepath)
        print(f"Model saved successfully to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from file"""
        if not torch.cuda.is_available():
            model_data = torch.load(filepath, map_location='cpu')
        else:
            model_data = torch.load(filepath)
        
        # Load configurations
        model_config = model_data['model_config']
        predictor_config = model_data['predictor_config']
        
        # Update predictor attributes
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
        
        print(f"Model loaded successfully from {filepath}")
        print(f"Model config: {model_config}")
        print(f"Predictor config: {predictor_config}")
    
    def predict_single(self, window_data):
        if len(window_data) < self.sequence_length:
            return 0.0
            
        features = window_data[self.feature_columns].values
        
        # No scaling - use raw features directly
        sequence = features[-self.sequence_length:]
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            prediction = self.model(X)
            
            if torch.isnan(prediction).any():
                return 0.0
        
        # Direct prediction value (no inverse transform needed)
        pred_value = prediction.cpu().numpy()[0][0]
        
        return max(0.0, pred_value) if not np.isnan(pred_value) else 0.0
    
    def evaluate_test_set(self, df, test_split=0.15):
        test_size = int(len(df) * test_split)
        test_data = df.tail(test_size).copy()
        
        predictions = []
        actuals = []
        timestamps = []
        
        for i in range(self.sequence_length, len(test_data)):
            try:
                window_data = test_data.iloc[i-self.sequence_length:i]
                pred = self.predict_single(window_data)
                actual = test_data.iloc[i]['arb_profit']
                timestamp = test_data.iloc[i]['datetime']
                
                if not np.isnan(pred) and not np.isnan(actual):
                    predictions.append(pred)
                    actuals.append(actual)
                    timestamps.append(timestamp)
            except:
                continue
        
        if len(predictions) == 0:
            print("No valid predictions generated")
            return
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Print results in requested format
        print("\nTest Results (Arbitrage Profit Prediction - RAW DATA VERSION):")
        print(f"{'DateTime':<12} {'Actual($)':<10} {'Predicted($)':<12} {'Error($)':<10} {'Error(%)':<8}")
        print("=" * 64)
        # help print the size of train and test data
        print(f"Test data size: {len(test_data)}")
        for i in range(min(20, len(predictions))):  # Show first 20 results
            dt = pd.to_datetime(timestamps[i]).strftime('%m-%d %H:%M')
            actual = actuals[i]
            pred = predictions[i]
            error = abs(actual - pred)
            error_pct = (error / (abs(actual) + 1e-8)) * 100
            
            print(f"{dt:<12} {actual:<10.4f} {pred:<12.4f} {error:<10.4f} {error_pct:<8.1f}")

        # Aggregate evaluation metrics
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        r2 = r2_score(actuals, predictions)

        print("\nSummary Metrics (Raw Data Model):")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R^2 : {r2:.4f}")
        
        print(f"\nTotal predictions evaluated: {len(predictions)}")
        print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"Actual range: [{actuals.min():.4f}, {actuals.max():.4f}]")


def main():
    predictor = ArbitragePredictor(sequence_length=200)
    
    # Load data
    try:
        df = predictor.load_data('/root/Research_Blog_Part_1/Data/final_data_task1_swell_CLEAN.csv')
    except FileNotFoundError:
        df = predictor.load_data('/root/Research_Blog_Part_1/Data/final_data_task1_swell.csv')
    
    print("Training Arbitrage Profit Prediction Model V3...")
    print("Version: RAW DATA (no clipping, no scaling)")
    print("Target: arb_profit (direct prediction from raw features)")
    print("Features include: prices, volume, delta_USD, ratios, moving averages")
    
    # Train
    test_size = int(len(df) * 0.15)
    train_val_data = df.iloc[:-test_size]
    
    predictor.train(train_val_data, epochs=30, batch_size=32, lr=0.001)
    
    # Test
    predictor.evaluate_test_set(df, test_split=0.1)
    
    # Save model with metadata
    predictor.save_model('arbitrage_profit_model_v3.pth')
    print("\nModel saved with metadata as 'arbitrage_profit_model_v3.pth'")

if __name__ == "__main__":
    main() 