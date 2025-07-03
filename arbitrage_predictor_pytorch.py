import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Additional imports for evaluation
try:
    import matplotlib.pyplot as plt
    plt.style.use('default')
except ImportError:
    print("Warning: matplotlib not available, plots will be skipped")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
        
        # More capacity for learning complex patterns
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Multi-layer FC with Tanh activations
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Proper initialization
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Multi-layer FC
        out = self.fc1(out)
        out = self.tanh1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        
        return out

class ArbitragePredictor:
    def __init__(self, sequence_length=200, prediction_horizon=1):
        """
        Long sequence length to capture extended patterns
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_scaler = StandardScaler()
        self.target_scaler = None  # Don't scale target initially
        self.model = None
        self.feature_columns = []
        
    def create_simple_features(self, df):
        """Create only the most important features"""
        df = df.copy()
        
        # Only the most basic and important features
        df['price_ratio'] = df['price_HYPE_HyperCORE'] / df['price_HYPE_HyperEVM']
        df['price_spread'] = df['price_HYPE_HyperCORE'] - df['price_HYPE_HyperEVM']
        
        # Very short-term moving averages only
        df['ma_ratio_3'] = df['price_ratio'].rolling(window=3).mean()
        df['ma_delta_3'] = df['delta_USD'].rolling(window=3).mean()
        
        # Simple momentum
        df['price_change'] = df['price_HYPE_HyperEVM'].pct_change()
        df['ratio_change'] = df['price_ratio'].pct_change()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
        
    def load_and_preprocess_data(self, file_path):
        """Load with minimal preprocessing"""
        print("Loading data...")
        df = pd.read_csv(file_path)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create simple features
        df = self.create_simple_features(df)
        
        # Select only key features
        base_features = ['price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'Amount_HYPE_HyperEVM']
        derived_features = ['price_ratio', 'price_spread', 'ma_ratio_3', 'ma_delta_3', 'price_change', 'ratio_change']
        
        self.feature_columns = base_features + derived_features
        
        print(f"Loaded {len(df)} records")
        print(f"Features ({len(self.feature_columns)}): {self.feature_columns}")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    
    def create_sequences(self, data, target_col='delta_USD'):
        """Sequence creation with outlier handling"""
        features = data[self.feature_columns].values
        target = data[target_col].values
        
        # Handle outliers in target - clip extreme values
        target_q95 = np.percentile(target, 95)
        target_clipped = np.clip(target, 0, target_q95)
        print(f"Target clipped at 95th percentile: {target_q95:.2f}")
        
        # Scale both features and target
        features_scaled = self.feature_scaler.fit_transform(features)
        self.target_scaler = StandardScaler()
        target_scaled = self.target_scaler.fit_transform(target_clipped.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:(i + self.sequence_length)])
            y.append(target_scaled[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, df, validation_split=0.2, epochs=150, batch_size=16, lr=0.001):
        """Training with better hyperparameters"""
        print("Preparing sequences...")
        X, y = self.create_sequences(df)
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Target statistics - Mean: {np.mean(y):.4f}, Std: {np.std(y):.4f}, Min: {np.min(y):.4f}, Max: {np.max(y):.4f}")
        
        # Handle NaN
        if np.isnan(X).any():
            X = np.nan_to_num(X)
        if np.isnan(y).any():
            y = np.nan_to_num(y)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Create datasets
        train_dataset = ArbitrageDataset(X_train, y_train)
        val_dataset = ArbitrageDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Better model with more capacity
        input_size = X.shape[2]
        self.model = LSTMArbitrageModel(input_size, hidden_size=128, num_layers=3, 
                                       output_size=1, dropout=0.2).to(device)
        
        # Better training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        print("Training improved model...")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 30
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_count = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Light gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()
                train_loss += loss.item()
                train_count += 1
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(1)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_count += 1
            
            train_loss = train_loss / train_count if train_count > 0 else float('inf')
            val_loss = val_loss / val_count if val_count > 0 else float('inf')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_arbitrage_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_arbitrage_model.pth'))
        
        return train_losses, val_losses
    
    def predict(self, df, last_n_points=None):
        """Predict with proper denormalization"""
        if last_n_points is None:
            last_n_points = self.sequence_length
            
        recent_data = df.tail(last_n_points)
        features = recent_data[self.feature_columns].values
        
        if np.isnan(features).any():
            features = np.nan_to_num(features)
        
        features_scaled = self.feature_scaler.transform(features)
        
        if np.isnan(features_scaled).any():
            features_scaled = np.nan_to_num(features_scaled)
        
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            
            if torch.isnan(prediction).any():
                return 0.0
        
        pred_scaled = prediction.cpu().numpy()[0][0]
        
        # Denormalize prediction
        if self.target_scaler is not None:
            pred_value = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred_value = pred_scaled
        
        if np.isnan(pred_value):
            return 0.0
        
        return max(0.0, pred_value)  # Ensure non-negative spread
    
    def calculate_arbitrage_score(self, df):
        """Tính toán arbitrage score cho ranking - dựa trên predicted spread"""
        scores = []
        
        for i in range(len(df)):
            if i < self.sequence_length:
                scores.append(0)
                continue
                
            # Lấy dữ liệu window
            window_data = df.iloc[i-self.sequence_length:i]
            predicted_spread = self.predict_on_subset(window_data)
            
            # Check for NaN in prediction
            if np.isnan(predicted_spread):
                scores.append(0)
                continue
            
            # Score dựa trên predicted spread và historical volatility
            historical_spreads = df['delta_USD'].iloc[max(0, i-20):i]
            spread_std = historical_spreads.std() if len(historical_spreads) > 1 else 1
            spread_mean = historical_spreads.mean() if len(historical_spreads) > 0 else 0
            
            # Check for NaN in statistics
            if np.isnan(spread_std) or np.isnan(spread_mean):
                scores.append(0)
                continue
            
            # Normalized score: (predicted - mean) / std
            score = (predicted_spread - spread_mean) / (spread_std + 1e-6)
            
            # Check for NaN in final score
            if np.isnan(score):
                scores.append(0)
            else:
                scores.append(max(0, score))  # Chỉ lấy score dương
        
        return scores
    
    def predict_on_subset(self, df):
        """Predict on a subset using existing scalers"""
        features = df[self.feature_columns].values
        
        if np.isnan(features).any():
            features = np.nan_to_num(features)
        
        features_scaled = self.feature_scaler.transform(features)
        
        if np.isnan(features_scaled).any():
            features_scaled = np.nan_to_num(features_scaled)
        
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X)
            
            if torch.isnan(prediction).any():
                return 0.0
        
        pred_scaled = prediction.cpu().numpy()[0][0]
        
        # Denormalize prediction
        if self.target_scaler is not None:
            pred_value = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
        else:
            pred_value = pred_scaled
        
        if np.isnan(pred_value):
            return 0.0
        
        return max(0.0, pred_value)  # Ensure non-negative spread
    
    def evaluate_on_test_set(self, df, test_split=0.15):
        """Đánh giá đơn giản trên test set"""
        print("\n=== TEST SET EVALUATION ===")
        
        # Split data and apply enhanced features
        test_size = int(len(df) * test_split)
        test_data = df.tail(test_size).copy()
        
        print(f"Test set size: {len(test_data)} records")
        
        # Collect predictions vs actuals
        predictions = []
        actuals = []
        
        for i in range(self.sequence_length, len(test_data)):
            try:
                window_data = test_data.iloc[i-self.sequence_length:i]
                pred = self.predict_on_subset(window_data)
                actual = test_data.iloc[i]['delta_USD']
                
                if not np.isnan(pred) and not np.isnan(actual):
                    predictions.append(pred)
                    actuals.append(actual)
            except:
                continue
        
        if len(predictions) == 0:
            print("Error: No valid predictions generated")
            return {}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Basic Metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        from sklearn.metrics import r2_score
        r2 = r2_score(actuals, predictions)
        correlation = np.corrcoef(actuals, predictions)[0, 1]
        
        print(f"Test Predictions: {len(predictions)}")
        print(f"MAE: ${mae:.4f}")
        print(f"RMSE: ${rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Correlation: {correlation:.4f}")
        print(f"Actual avg: ${np.mean(actuals):.4f}, Pred avg: ${np.mean(predictions):.4f}")
        
        # Tạo bảng so sánh actual vs predicted
        self.create_comparison_table(actuals, predictions, test_data)
        
        return {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'correlation': correlation, 'num_predictions': len(predictions)
        }
    
    def create_comparison_table(self, actuals, predictions, test_data):
        """Tạo bảng so sánh actual vs predicted spreads"""
        print(f"\n=== ACTUAL vs PREDICTED SPREAD COMPARISON ===")
        
        # Lấy 20 samples đầu và 20 samples cuối để hiển thị
        n_samples = min(20, len(actuals))
        
        # Tạo dataframe để dễ format
        import pandas as pd
        
        # Lấy timestamps tương ứng
        start_idx = self.sequence_length
        timestamps = test_data['datetime'].iloc[start_idx:start_idx+len(actuals)].values
        
        # Tạo comparison table
        comparison_df = pd.DataFrame({
            'DateTime': timestamps,
            'Actual_Spread': actuals,
            'Predicted_Spread': predictions,
            'Error': np.abs(actuals - predictions),
            'Error_Pct': np.abs(actuals - predictions) / (np.abs(actuals) + 1e-8) * 100
        })
        
        # Format datetime
        comparison_df['DateTime'] = pd.to_datetime(comparison_df['DateTime']).dt.strftime('%m-%d %H:%M')
        
        print(f"First {n_samples} predictions:")
        print("=" * 80)
        print(f"{'DateTime':<12} {'Actual($)':<10} {'Predicted($)':<12} {'Error($)':<10} {'Error(%)':<8}")
        print("=" * 80)
        
        for i in range(n_samples):
            dt = comparison_df.iloc[i]['DateTime']
            actual = comparison_df.iloc[i]['Actual_Spread']
            pred = comparison_df.iloc[i]['Predicted_Spread']
            error = comparison_df.iloc[i]['Error']
            error_pct = comparison_df.iloc[i]['Error_Pct']
            
            print(f"{dt:<12} {actual:<10.4f} {pred:<12.4f} {error:<10.4f} {error_pct:<8.1f}")
        
        if len(actuals) > n_samples * 2:
            print("..." + " " * 70)
            print(f"Last {n_samples} predictions:")
            print("=" * 80)
            print(f"{'DateTime':<12} {'Actual($)':<10} {'Predicted($)':<12} {'Error($)':<10} {'Error(%)':<8}")
            print("=" * 80)
            
            for i in range(len(actuals) - n_samples, len(actuals)):
                dt = comparison_df.iloc[i]['DateTime']
                actual = comparison_df.iloc[i]['Actual_Spread']
                pred = comparison_df.iloc[i]['Predicted_Spread']
                error = comparison_df.iloc[i]['Error']
                error_pct = comparison_df.iloc[i]['Error_Pct']
                
                print(f"{dt:<12} {actual:<10.4f} {pred:<12.4f} {error:<10.4f} {error_pct:<8.1f}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS:")
        print(f"Best Prediction  (lowest error):  Actual: ${comparison_df.loc[comparison_df['Error'].idxmin(), 'Actual_Spread']:.4f}, Predicted: ${comparison_df.loc[comparison_df['Error'].idxmin(), 'Predicted_Spread']:.4f}")
        print(f"Worst Prediction (highest error): Actual: ${comparison_df.loc[comparison_df['Error'].idxmax(), 'Actual_Spread']:.4f}, Predicted: ${comparison_df.loc[comparison_df['Error'].idxmax(), 'Predicted_Spread']:.4f}")
        print(f"Average Error: ${comparison_df['Error'].mean():.4f}")
        print(f"Median Error:  ${comparison_df['Error'].median():.4f}")
        
        # Lưu full table ra file
        comparison_df.to_csv('actual_vs_predicted_spreads.csv', index=False)
        print(f"\nFull comparison table saved to 'actual_vs_predicted_spreads.csv'")
        
        return comparison_df
    


def main():
    # Khởi tạo predictor với scaled up model
    predictor = ArbitragePredictor(sequence_length=200, prediction_horizon=1)
    
    # Load dữ liệu
    df = predictor.load_and_preprocess_data('final_data_task1_swell.csv')
    
    print("\nData statistics:")
    print(df[['arb_profit', 'delta_USD', 'price_ratio']].describe())
    
    # Training với proper data split
    print("\n=== TRAINING SCALED LSTM MODEL ===")
    test_size = int(len(df) * 0.15)
    train_val_data = df.iloc[:-test_size]
    
    print(f"Training on {len(train_val_data)} records, Testing on {test_size} records")
    print(f"Model: 3-layer LSTM (128 hidden units) with Tanh activation")
    print(f"Sequence length: {predictor.sequence_length}")
    print(f"Features ({len(predictor.feature_columns)}): {predictor.feature_columns}")
    print("Starting training...")
    
    train_losses, val_losses = predictor.train(train_val_data, epochs=150, batch_size=16, lr=0.001)
    
    # TEST SET EVALUATION
    test_results = predictor.evaluate_on_test_set(df, test_split=0.15)
    
    # Prediction for next arbitrage
    print("\n=== NEXT SPREAD PREDICTION ===")
    next_spread_prediction = predictor.predict_on_subset(df.tail(predictor.sequence_length))
    print(f"Predicted spread for next arbitrage: ${next_spread_prediction:.4f}")
    
    recent_spreads = df['delta_USD'].tail(10)
    print(f"Recent actual spreads avg: ${recent_spreads.mean():.4f}")
    
    # Calculate potential profit with average amount
    avg_amount = df['Amount_HYPE_HyperEVM'].tail(10).mean()
    potential_profit = next_spread_prediction * avg_amount
    print(f"Potential profit with avg amount ({avg_amount:.2f}): ${potential_profit:.4f}")
    
    # Final Summary
    if test_results:
        print(f"\n=== FINAL MODEL PERFORMANCE ===")
        print(f"Test MAE: ${test_results['mae']:.4f}")
        print(f"Test R²: {test_results['r2']:.4f}")
        print(f"Test Correlation: {test_results['correlation']:.4f}")
        
        quality = "GOOD" if test_results['r2'] > 0.5 else "MODERATE" if test_results['r2'] > 0.3 else "NEEDS IMPROVEMENT"
        print(f"Model Quality: {quality}")
    
    # Save model
    torch.save(predictor.model.state_dict(), 'scaled_arbitrage_predictor.pth')
    print(f"\nModel saved as 'scaled_arbitrage_predictor.pth'")

if __name__ == "__main__":
    main() 