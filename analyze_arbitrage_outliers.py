import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_analyze_data(file_path):
    """Load CSV data and perform basic analysis"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert datetime columns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['time'] = pd.to_datetime(df['time'])
    
    return df

def analyze_arb_profit_distribution(df):
    """Analyze the distribution of arb_profit"""
    arb_profit = df['arb_profit']
    
    print("\n=== ARB_PROFIT DISTRIBUTION ANALYSIS ===")
    print(f"Number of records: {len(arb_profit)}")
    print(f"Min: {arb_profit.min():.6f}")
    print(f"Max: {arb_profit.max():.6f}")
    print(f"Mean: {arb_profit.mean():.6f}")
    print(f"Median: {arb_profit.median():.6f}")
    print(f"Std: {arb_profit.std():.6f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    print("\nPercentiles:")
    for p in percentiles:
        value = np.percentile(arb_profit, p)
        print(f"P{p}: {value:.6f}")
    
    return arb_profit

def identify_outliers_iqr(df, multiplier=1.5):
    """Identify outliers using IQR method"""
    arb_profit = df['arb_profit']
    
    Q1 = arb_profit.quantile(0.25)
    Q3 = arb_profit.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers_mask = (arb_profit < lower_bound) | (arb_profit > upper_bound)
    outliers = df[outliers_mask]
    
    print(f"\n=== OUTLIERS DETECTION (IQR method, multiplier={multiplier}) ===")
    print(f"Q1: {Q1:.6f}")
    print(f"Q3: {Q3:.6f}")
    print(f"IQR: {IQR:.6f}")
    print(f"Lower bound: {lower_bound:.6f}")
    print(f"Upper bound: {upper_bound:.6f}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers, upper_bound, lower_bound

def identify_outliers_zscore(df, threshold=3):
    """Identify outliers using Z-score method"""
    arb_profit = df['arb_profit']
    
    z_scores = np.abs(stats.zscore(arb_profit))
    outliers_mask = z_scores > threshold
    outliers = df[outliers_mask]
    
    print(f"\n=== OUTLIERS DETECTION (Z-score method, threshold={threshold}) ===")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers

def identify_outliers_percentile(df, percentile=95):
    """Identify outliers using percentile method"""
    arb_profit = df['arb_profit']
    
    threshold = np.percentile(arb_profit, percentile)
    outliers_mask = arb_profit > threshold
    outliers = df[outliers_mask]
    
    print(f"\n=== OUTLIERS DETECTION (Percentile method, P{percentile}) ===")
    print(f"Threshold: {threshold:.6f}")
    print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers, threshold

def save_outliers_to_csv(outliers, method_name, output_file):
    """Save outliers to CSV file"""
    # Sort by arb_profit descending
    outliers_sorted = outliers.sort_values('arb_profit', ascending=False)
    
    outliers_sorted.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(outliers_sorted)} outliers to file: {output_file}")
    
    # Show top 10 highest arb_profit
    print(f"\n=== TOP 10 HIGHEST ARB_PROFIT ({method_name}) ===")
    top_10 = outliers_sorted.head(10)
    for idx, row in top_10.iterrows():
        print(f"Rank {len(top_10) - list(top_10.index).index(idx)}: "
              f"arb_profit={row['arb_profit']:.6f}, "
              f"amount={row['Amount_HYPE_HyperEVM']:.2f}, "
              f"delta_USD={row['delta_USD']:.6f}, "
              f"datetime={row['datetime']}")

def create_visualization(df, outliers, output_image):
    """Create visualization of arb_profit distribution and outliers"""
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Histogram
    plt.subplot(2, 3, 1)
    plt.hist(df['arb_profit'], bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Arb Profit')
    plt.ylabel('Frequency')
    plt.title('Distribution of Arb Profit')
    plt.yscale('log')
    
    # Subplot 2: Box plot
    plt.subplot(2, 3, 2)
    plt.boxplot(df['arb_profit'])
    plt.ylabel('Arb Profit')
    plt.title('Box Plot of Arb Profit')
    
    # Subplot 3: Scatter plot of outliers
    plt.subplot(2, 3, 3)
    plt.scatter(range(len(outliers)), outliers['arb_profit'], 
                c='red', alpha=0.6, s=20)
    plt.xlabel('Outlier Index')
    plt.ylabel('Arb Profit')
    plt.title('Outliers Scatter Plot')
    
    # Subplot 4: Time series of arb_profit
    plt.subplot(2, 3, 4)
    plt.plot(df['datetime'], df['arb_profit'], alpha=0.5, linewidth=0.5)
    plt.scatter(outliers['datetime'], outliers['arb_profit'], 
                c='red', alpha=0.8, s=15, label='Outliers')
    plt.xlabel('DateTime')
    plt.ylabel('Arb Profit')
    plt.title('Arb Profit Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Subplot 5: Correlation with amount
    plt.subplot(2, 3, 5)
    plt.scatter(df['Amount_HYPE_HyperEVM'], df['arb_profit'], 
                alpha=0.5, s=10, label='All data')
    plt.scatter(outliers['Amount_HYPE_HyperEVM'], outliers['arb_profit'], 
                c='red', alpha=0.8, s=15, label='Outliers')
    plt.xlabel('Amount HYPE HyperEVM')
    plt.ylabel('Arb Profit')
    plt.title('Arb Profit vs Amount')
    plt.legend()
    
    # Subplot 6: Correlation with delta_USD
    plt.subplot(2, 3, 6)
    plt.scatter(df['delta_USD'], df['arb_profit'], 
                alpha=0.5, s=10, label='All data')
    plt.scatter(outliers['delta_USD'], outliers['arb_profit'], 
                c='red', alpha=0.8, s=15, label='Outliers')
    plt.xlabel('Delta USD')
    plt.ylabel('Arb Profit')
    plt.title('Arb Profit vs Delta USD')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Saved chart to file: {output_image}")

def main():
    input_file = "/root/Research_Blog_Part_1/2025_03_22_ARB_EVM_CORE_DATA.csv"
    
    print("🔍 ARBITRAGE PROFIT OUTLIERS ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = load_and_analyze_data(input_file)
    
    # Analyze distribution
    arb_profit = analyze_arb_profit_distribution(df)
    
    # Method 1: IQR method (conservative)
    outliers_iqr, upper_bound_iqr, lower_bound_iqr = identify_outliers_iqr(df, multiplier=1.5)
    save_outliers_to_csv(outliers_iqr, "IQR Method", "outliers_iqr_conservative.csv")
    
    # Method 2: IQR method (aggressive)
    outliers_iqr_agg, _, _ = identify_outliers_iqr(df, multiplier=1.0)
    save_outliers_to_csv(outliers_iqr_agg, "IQR Aggressive", "outliers_iqr_aggressive.csv")
    
    # Method 3: Z-score method
    outliers_zscore = identify_outliers_zscore(df, threshold=2.5)
    save_outliers_to_csv(outliers_zscore, "Z-score Method", "outliers_zscore.csv")
    
    # Method 4: Percentile method (top 5%)
    outliers_p95, threshold_p95 = identify_outliers_percentile(df, percentile=95)
    save_outliers_to_csv(outliers_p95, "Top 5%", "outliers_top5_percent.csv")
    
    # Method 5: Percentile method (top 1%)
    outliers_p99, threshold_p99 = identify_outliers_percentile(df, percentile=99)
    save_outliers_to_csv(outliers_p99, "Top 1%", "outliers_top1_percent.csv")
    
    # Create visualization
    create_visualization(df, outliers_p95, "arb_profit_analysis.png")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY OF RESULTS:")
    print(f"• Total records: {len(df)}")
    print(f"• Outliers (IQR conservative): {len(outliers_iqr)} records")
    print(f"• Outliers (IQR aggressive): {len(outliers_iqr_agg)} records")
    print(f"• Outliers (Z-score): {len(outliers_zscore)} records")
    print(f"• Outliers (Top 5%): {len(outliers_p95)} records")
    print(f"• Outliers (Top 1%): {len(outliers_p99)} records")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"• Use file 'outliers_top5_percent.csv' to analyze transactions with highest profits")
    print(f"• Use file 'outliers_iqr_conservative.csv' to analyze outliers using standard statistical method")
    print(f"• Check file 'arb_profit_analysis.png' to view the analysis chart")

if __name__ == "__main__":
    main() 