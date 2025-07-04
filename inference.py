import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Check if we have enough data
        if len(df) < self.sequence_length:
            print(f"Warning: Dataset has {len(df)} rows but model requires {self.sequence_length} rows for sequence length.")
            print("Attempting to make predictions with available data...")
            
            # Try to make predictions with available data by padding
            predictions = []
            timestamps = []
            
            for i in range(len(df)):
                try:
                    # Use all available data up to current point
                    window_data = df.iloc[:i+1]
                    
                    # Pad with zeros if needed
                    if len(window_data) < self.sequence_length:
                        padding_rows = self.sequence_length - len(window_data)
                        padding_data = pd.DataFrame(0, index=range(padding_rows), columns=window_data.columns)
                        window_data = pd.concat([padding_data, window_data], ignore_index=True)
                    
                    pred = self.predict_single(window_data)
                    timestamp = df.iloc[i]['datetime'] if 'datetime' in df.columns else i
                    
                    predictions.append(pred)
                    timestamps.append(timestamp)
                except Exception as e:
                    print(f"Error predicting at index {i}: {e}")
                    predictions.append(0.0)
                    timestamps.append(i)
            
            return predictions, timestamps
        
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
    
    def calculate_accuracy_metrics(self, actual_profits, predicted_profits):
        """Calculate various accuracy metrics"""
        # Remove any NaN values
        mask = ~(np.isnan(actual_profits) | np.isnan(predicted_profits))
        actual_clean = actual_profits[mask]
        predicted_clean = predicted_profits[mask]
        
        if len(actual_clean) == 0:
            return {
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'correlation': np.nan
            }
        
        # Calculate metrics
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_clean, predicted_clean)
        r2 = r2_score(actual_clean, predicted_clean)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual_clean - predicted_clean) / (actual_clean + 1e-8))) * 100
        
        # Correlation coefficient
        correlation = np.corrcoef(actual_clean, predicted_clean)[0, 1]
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'correlation': correlation
        }
    
    def print_accuracy_report(self, actual_profits, predicted_profits):
        """Print a comprehensive accuracy report"""
        metrics = self.calculate_accuracy_metrics(actual_profits, predicted_profits)
        
        print("\n" + "="*60)
        print("ACCURACY REPORT")
        print("="*60)
        print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"R-squared (R²): {metrics['r2']:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
        print(f"Correlation Coefficient: {metrics['correlation']:.6f}")
        print("="*60)
        
        # Additional insights
        if not np.isnan(metrics['r2']):
            if metrics['r2'] >= 0.8:
                print("✓ Excellent model performance (R² ≥ 0.8)")
            elif metrics['r2'] >= 0.6:
                print("✓ Good model performance (R² ≥ 0.6)")
            elif metrics['r2'] >= 0.4:
                print("~ Moderate model performance (R² ≥ 0.4)")
            else:
                print("✗ Poor model performance (R² < 0.4)")
        
        if not np.isnan(metrics['correlation']):
            if abs(metrics['correlation']) >= 0.8:
                print("✓ Strong correlation between actual and predicted values")
            elif abs(metrics['correlation']) >= 0.6:
                print("✓ Moderate correlation between actual and predicted values")
            else:
                print("~ Weak correlation between actual and predicted values")
        
        return metrics

    def create_visualization_plots(self, results_df, save_path=None):
        """Create comprehensive visualization plots for the results"""
        if 'actual_arbitrage_profit' not in results_df.columns:
            print("No actual arbitrage profit data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Arbitrage Profit Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted scatter plot
        ax1 = axes[0, 0]
        actual = results_df['actual_arbitrage_profit']
        predicted = results_df['predicted_arbitrage_profit']
        
        ax1.scatter(actual, predicted, alpha=0.6, s=30)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Arbitrage Profit ($)')
        ax1.set_ylabel('Predicted Arbitrage Profit ($)')
        ax1.set_title('Actual vs Predicted Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² value to the plot
        r2 = r2_score(actual, predicted)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 2: Time series of actual vs predicted
        ax2 = axes[0, 1]
        timestamps = pd.to_datetime(results_df['timestamp'])
        ax2.plot(timestamps, actual, label='Actual', alpha=0.7, linewidth=1)
        ax2.plot(timestamps, predicted, label='Predicted', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Arbitrage Profit ($)')
        ax2.set_title('Time Series: Actual vs Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Residuals plot
        ax3 = axes[1, 0]
        residuals = actual - predicted
        ax3.scatter(predicted, residuals, alpha=0.6, s=30)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals (Actual - Predicted)')
        ax3.set_title('Residuals Plot')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution comparison
        ax4 = axes[1, 1]
        ax4.hist(actual, bins=30, alpha=0.7, label='Actual', density=True)
        ax4.hist(predicted, bins=30, alpha=0.7, label='Predicted', density=True)
        ax4.set_xlabel('Arbitrage Profit ($)')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        # Create additional summary statistics plot
        self.create_summary_statistics_plot(results_df, save_path.replace('.png', '_summary.png') if save_path else None)
    
    def create_summary_statistics_plot(self, results_df, save_path=None):
        """Create a summary statistics visualization"""
        if 'actual_arbitrage_profit' not in results_df.columns:
            return
        
        actual = results_df['actual_arbitrage_profit']
        predicted = results_df['predicted_arbitrage_profit']
        
        # Calculate metrics
        metrics = self.calculate_accuracy_metrics(actual, predicted)
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Metrics comparison
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)']
        metric_values = [metrics['mse'], metrics['rmse'], metrics['mae'], 
                        metrics['r2'], metrics['mape']]
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax1.set_title('Accuracy Metrics')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Statistical summary
        stats_data = {
            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Median'],
            'Actual': [actual.mean(), actual.std(), actual.min(), actual.max(), actual.median()],
            'Predicted': [predicted.mean(), predicted.std(), predicted.min(), predicted.max(), predicted.median()]
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_df_melted = stats_df.melt(id_vars=['Metric'], var_name='Type', value_name='Value')
        
        sns.barplot(data=stats_df_melted, x='Metric', y='Value', hue='Type', ax=ax2)
        ax2.set_title('Statistical Summary')
        ax2.set_ylabel('Value ($)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary statistics plot saved to {save_path}")
        
        plt.show()

    def predict_from_csv(self, csv_path, output_path=None):
        """Load data from CSV and make predictions"""
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset loaded with {len(df)} rows")
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create features
        df = self.create_features(df)
        print(f"Features created. Available columns: {list(df.columns)}")
        
        # Make predictions
        predictions, timestamps = self.predict_batch(df)
        print(f"Generated {len(predictions)} predictions")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_arbitrage_profit': predictions
        })
        
        # Add original data if available
        if len(df) >= len(predictions):
            # For small datasets, use all available data
            if len(df) < self.sequence_length:
                original_data = df.iloc[:len(predictions)]
            else:
                original_data = df.iloc[self.sequence_length:self.sequence_length+len(predictions)]
            
            for col in ['price_HYPE_HyperEVM', 'price_HYPE_HyperCORE', 'Amount_HYPE_HyperEVM', 'delta_USD']:
                if col in original_data.columns:
                    results_df[f'original_{col}'] = original_data[col].values
            
            # Add actual arbitrage profit if available
            if 'arb_profit' in original_data.columns:
                results_df['actual_arbitrage_profit'] = original_data['arb_profit'].values
                
                # Calculate and print accuracy metrics
                actual_profits = results_df['actual_arbitrage_profit'].values
                predicted_profits = results_df['predicted_arbitrage_profit'].values
                
                self.print_accuracy_report(actual_profits, predicted_profits)
                
                # Create visualization plots
                plot_path = output_path.replace('.csv', '_plots.png') if output_path else 'prediction_plots.png'
                self.create_visualization_plots(results_df, plot_path)
        
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
            
            # If actual profits are available, show comparison
            if 'actual_arbitrage_profit' in results.columns:
                print("\nActual vs Predicted Comparison (First 10):")
                comparison_df = results[['timestamp', 'actual_arbitrage_profit', 'predicted_arbitrage_profit']].head(10)
                print(comparison_df.to_string(index=False))
            
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