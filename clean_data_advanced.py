import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_data(file_path):
    """Load CSV data"""
    print("📂 Loading data...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert datetime columns
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['time'] = pd.to_datetime(df['time'])
    
    return df

def analyze_arb_profit_stats(df):
    """Analyze arb_profit statistics"""
    arb_profit = df['arb_profit']
    
    print("\n📊 ARB_PROFIT ANALYSIS:")
    print(f"• Number of records: {len(arb_profit):,}")
    print(f"• Min: {arb_profit.min():.6f}")
    print(f"• Max: {arb_profit.max():.6f}")
    print(f"• Mean: {arb_profit.mean():.6f}")
    print(f"• Median: {arb_profit.median():.6f}")
    print(f"• Std: {arb_profit.std():.6f}")
    
    # Key percentiles for understanding distribution
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("\n📈 Percentiles:")
    for p in percentiles:
        value = np.percentile(arb_profit, p)
        print(f"  P{p}: {value:.6f}")
    
    # Analysis of very small profits
    small_profits = arb_profit[arb_profit < 0.1]
    tiny_profits = arb_profit[arb_profit < 0.01]
    
    print(f"\n💰 SMALL PROFITS ANALYSIS:")
    print(f"• Profits < 0.1: {len(small_profits):,} ({len(small_profits)/len(arb_profit)*100:.2f}%)")
    print(f"• Profits < 0.01: {len(tiny_profits):,} ({len(tiny_profits)/len(arb_profit)*100:.2f}%)")
    
    return arb_profit

def estimate_transaction_fees():
    """Estimate typical transaction fees for arbitrage"""
    print("\n💸 TRANSACTION FEES ESTIMATION:")
    print("• DEX swap fee: ~0.3% per transaction")
    print("• Bridge/transfer fee: ~$1-5")
    print("• Gas fees: ~$0.5-2")
    print("• Slippage: ~0.1-0.5%")
    print("• Total estimate: $2-10 + ~0.4-0.8% of volume")
    
    # For a typical arbitrage, minimum profit should cover these costs
    # Assuming average transaction size and fees
    min_profit_suggestion = 0.05  # $0.05 minimum
    percentage_fee = 0.008  # 0.8% total fees
    
    print(f"\n🎯 MINIMUM THRESHOLD RECOMMENDATIONS:")
    print(f"• Absolute profit: >= ${min_profit_suggestion}")
    print(f"• Or >= {percentage_fee*100}% of transaction volume")
    
    return min_profit_suggestion, percentage_fee

def determine_cleaning_thresholds(df):
    """Determine both upper and lower thresholds for cleaning"""
    arb_profit = df['arb_profit']
    
    # Upper threshold: Remove extreme outliers using IQR method
    Q1 = arb_profit.quantile(0.25)
    Q3 = arb_profit.quantile(0.75)
    IQR = Q3 - Q1
    upper_threshold = Q3 + 1.5 * IQR
    
    # Lower threshold: Remove profits below $0.5 (aggressive filtering)
    lower_threshold = 0.5  # Fixed threshold: $0.5 minimum
    
    print(f"\n🎯 CLEANING THRESHOLDS:")
    print(f"• Remove arb_profit > {upper_threshold:.6f} (high outliers)")
    print(f"• Remove arb_profit < {lower_threshold:.6f} (below $0.5 - too small)")
    print(f"• Keep: {lower_threshold:.6f} <= arb_profit <= {upper_threshold:.6f}")
    
    return lower_threshold, upper_threshold

def clean_data_comprehensive(df, lower_threshold, upper_threshold):
    """Remove both high outliers and unprofitable small transactions"""
    original_count = len(df)
    
    # Identify records to remove
    too_high_mask = df['arb_profit'] > upper_threshold
    too_low_mask = df['arb_profit'] < lower_threshold
    
    high_outliers = df[too_high_mask].copy()
    low_outliers = df[too_low_mask].copy()
    
    # Clean data (keep only profitable, reasonable transactions)
    clean_mask = (df['arb_profit'] >= lower_threshold) & (df['arb_profit'] <= upper_threshold)
    clean_df = df[clean_mask].copy()
    
    # Statistics
    high_removed = len(high_outliers)
    low_removed = len(low_outliers)
    total_removed = high_removed + low_removed
    clean_count = len(clean_df)
    
    print(f"\n🧹 COMPREHENSIVE CLEANING RESULTS:")
    print(f"• Original records: {original_count:,}")
    print(f"• Removed (too high): {high_removed:,} ({high_removed/original_count*100:.2f}%)")
    print(f"• Removed (too low): {low_removed:,} ({low_removed/original_count*100:.2f}%)")
    print(f"• Total removed: {total_removed:,} ({total_removed/original_count*100:.2f}%)")
    print(f"• Clean records: {clean_count:,} ({clean_count/original_count*100:.2f}%)")
    
    # Analyze removed data
    if len(high_outliers) > 0:
        print(f"\n📊 REMOVED HIGH OUTLIERS:")
        print(f"• Min: {high_outliers['arb_profit'].min():.6f}")
        print(f"• Max: {high_outliers['arb_profit'].max():.6f}")
        print(f"• Mean: {high_outliers['arb_profit'].mean():.6f}")
        
        # Top 3 highest
        top_high = high_outliers.nlargest(3, 'arb_profit')
        print(f"• Top 3 highest:")
        for i, (_, row) in enumerate(top_high.iterrows(), 1):
            print(f"  {i}. {row['arb_profit']:.6f} (amount: {row['Amount_HYPE_HyperEVM']:.2f})")
    
    if len(low_outliers) > 0:
        print(f"\n📊 REMOVED LOW PROFITS:")
        print(f"• Min: {low_outliers['arb_profit'].min():.6f}")
        print(f"• Max: {low_outliers['arb_profit'].max():.6f}")
        print(f"• Mean: {low_outliers['arb_profit'].mean():.6f}")
        print(f"• Reason: Below $0.5 threshold (too small to be meaningful)")
    
    return clean_df, high_outliers, low_outliers

def analyze_clean_data(clean_df):
    """Analyze the cleaned data"""
    arb_profit = clean_df['arb_profit']
    
    print(f"\n📊 CLEANED DATA ANALYSIS:")
    print(f"• Number of records: {len(arb_profit):,}")
    print(f"• Min: {arb_profit.min():.6f}")
    print(f"• Max: {arb_profit.max():.6f}")
    print(f"• Mean: {arb_profit.mean():.6f}")
    print(f"• Median: {arb_profit.median():.6f}")
    print(f"• Std: {arb_profit.std():.6f}")
    
    # Profit distribution in clean data
    profit_ranges = [
        (0.02, 0.05, "Small"),
        (0.05, 0.1, "Medium"), 
        (0.1, 0.5, "Good"),
        (0.5, 1.0, "High"),
        (1.0, float('inf'), "Very High")
    ]
    
    print(f"\n💰 PROFIT DISTRIBUTION AFTER CLEANING:")
    for min_val, max_val, label in profit_ranges:
        if max_val == float('inf'):
            count = len(arb_profit[arb_profit >= min_val])
        else:
            count = len(arb_profit[(arb_profit >= min_val) & (arb_profit < max_val)])
        
        percentage = count / len(arb_profit) * 100
        print(f"• {label} (${min_val}-{max_val if max_val != float('inf') else '∞'}): {count:,} ({percentage:.1f}%)")

def create_comprehensive_visualization(original_df, clean_df, high_outliers, low_outliers, output_file):
    """Create comprehensive visualization of the cleaning process"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # Plot 1: Original distribution
    axes[0, 0].hist(original_df['arb_profit'], bins=100, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_xlabel('Arb Profit')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Original Data Distribution')
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Clean distribution
    axes[0, 1].hist(clean_df['arb_profit'], bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Arb Profit')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Clean Data Distribution')
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Removed data
    removed_data = []
    if len(high_outliers) > 0:
        removed_data.extend(high_outliers['arb_profit'].tolist())
    if len(low_outliers) > 0:
        removed_data.extend(low_outliers['arb_profit'].tolist())
    
    if removed_data:
        axes[0, 2].hist(removed_data, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 2].set_xlabel('Arb Profit')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('All Removed Data')
        axes[0, 2].set_yscale('log')
    
    # Plot 4: Box plot comparison
    box_data = [original_df['arb_profit'], clean_df['arb_profit']]
    box_labels = ['Original', 'Clean']
    axes[1, 0].boxplot(box_data, labels=box_labels)
    axes[1, 0].set_ylabel('Arb Profit')
    axes[1, 0].set_title('Box Plot Comparison')
    axes[1, 0].set_yscale('log')
    
    # Plot 5: Time series
    axes[1, 1].plot(original_df['datetime'], original_df['arb_profit'], 
                    alpha=0.3, linewidth=0.5, color='blue', label='Original')
    axes[1, 1].plot(clean_df['datetime'], clean_df['arb_profit'], 
                    alpha=0.8, linewidth=0.8, color='green', label='Clean')
    if len(high_outliers) > 0:
        axes[1, 1].scatter(high_outliers['datetime'], high_outliers['arb_profit'], 
                          c='red', alpha=0.7, s=20, label='High outliers')
    if len(low_outliers) > 0:
        axes[1, 1].scatter(low_outliers['datetime'], low_outliers['arb_profit'], 
                          c='orange', alpha=0.7, s=10, label='Low profits')
    axes[1, 1].set_xlabel('DateTime')
    axes[1, 1].set_ylabel('Arb Profit')
    axes[1, 1].set_title('Time Series Comparison')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Profit vs Volume
    axes[1, 2].scatter(clean_df['Amount_HYPE_HyperEVM'], clean_df['arb_profit'], 
                       alpha=0.6, s=15, color='green', label='Clean data')
    if len(high_outliers) > 0:
        axes[1, 2].scatter(high_outliers['Amount_HYPE_HyperEVM'], high_outliers['arb_profit'], 
                          c='red', alpha=0.7, s=20, label='High outliers')
    if len(low_outliers) > 0:
        axes[1, 2].scatter(low_outliers['Amount_HYPE_HyperEVM'], low_outliers['arb_profit'], 
                          c='orange', alpha=0.7, s=15, label='Low profits')
    axes[1, 2].set_xlabel('Amount HYPE HyperEVM')
    axes[1, 2].set_ylabel('Arb Profit')
    axes[1, 2].set_title('Profit vs Volume')
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')
    
    # Plot 7: Clean data profit distribution (detailed)
    axes[2, 0].hist(clean_df['arb_profit'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2, 0].set_xlabel('Arb Profit')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Clean Data - Detailed Distribution')
    
    # Plot 8: Statistics comparison
    stats_comparison = {
        'Metric': ['Count', 'Mean', 'Median', 'Std'],
        'Original': [
            len(original_df),
            original_df['arb_profit'].mean(),
            original_df['arb_profit'].median(),
            original_df['arb_profit'].std()
        ],
        'Clean': [
            len(clean_df),
            clean_df['arb_profit'].mean(),
            clean_df['arb_profit'].median(),
            clean_df['arb_profit'].std()
        ]
    }
    
    x = np.arange(len(stats_comparison['Metric']))
    width = 0.35
    
    axes[2, 1].bar(x - width/2, stats_comparison['Original'], width, 
                   label='Original', alpha=0.7, color='lightblue')
    axes[2, 1].bar(x + width/2, stats_comparison['Clean'], width, 
                   label='Clean', alpha=0.7, color='lightgreen')
    axes[2, 1].set_xlabel('Metrics')
    axes[2, 1].set_ylabel('Values')
    axes[2, 1].set_title('Statistics Comparison')
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(stats_comparison['Metric'])
    axes[2, 1].legend()
    axes[2, 1].set_yscale('log')
    
    # Plot 9: Cleaning summary
    cleaning_data = ['Original', 'High outliers', 'Low profits', 'Clean']
    cleaning_counts = [
        len(original_df),
        len(high_outliers) if len(high_outliers) > 0 else 0,
        len(low_outliers) if len(low_outliers) > 0 else 0,
        len(clean_df)
    ]
    colors = ['lightblue', 'red', 'orange', 'lightgreen']
    
    axes[2, 2].bar(cleaning_data, cleaning_counts, color=colors, alpha=0.7)
    axes[2, 2].set_ylabel('Number of Records')
    axes[2, 2].set_title('Data Cleaning Summary')
    axes[2, 2].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for i, count in enumerate(cleaning_counts):
        axes[2, 2].text(i, count + max(cleaning_counts)*0.01, f'{count:,}', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Saved comprehensive analysis chart to: {output_file}")

def main():
    input_file = "/root/Research_Blog_Part_1/2025_03_22_ARB_EVM_CORE_DATA.csv"
    clean_output_file = "final_data_task1_swell_CLEAN.csv"
    high_outliers_file = "removed_high_outliers.csv"
    low_profits_file = "removed_low_profits.csv"
    plot_output_file = "comprehensive_data_cleaning.png"
    
    print("🧹 COMPREHENSIVE DATA CLEANING - REMOVE OUTLIERS & LOW PROFITS")
    print("=" * 70)
    
    # Load original data
    df = load_data(input_file)
    
    # Analyze original data
    analyze_arb_profit_stats(df)
    
    # Determine cleaning thresholds
    lower_threshold, upper_threshold = determine_cleaning_thresholds(df)
    
    # Clean data comprehensively
    clean_df, high_outliers, low_outliers = clean_data_comprehensive(df, lower_threshold, upper_threshold)
    
    # Analyze clean data
    analyze_clean_data(clean_df)
    
    # Save all results
    clean_df.to_csv(clean_output_file, index=False)
    print(f"\n💾 SAVED CLEAN DATA: {clean_output_file}")
    
    if len(high_outliers) > 0:
        high_outliers_sorted = high_outliers.sort_values('arb_profit', ascending=False)
        high_outliers_sorted.to_csv(high_outliers_file, index=False)
        print(f"💾 SAVED HIGH OUTLIERS: {high_outliers_file}")
    
    if len(low_outliers) > 0:
        low_outliers_sorted = low_outliers.sort_values('arb_profit', ascending=True)
        low_outliers_sorted.to_csv(low_profits_file, index=False)
        print(f"💾 SAVED LOW PROFITS: {low_profits_file}")
    
    # Create comprehensive visualization
    create_comprehensive_visualization(df, clean_df, high_outliers, low_outliers, plot_output_file)
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 COMPREHENSIVE CLEANING RESULTS SUMMARY:")
    print(f"• Lower threshold: arb_profit >= $0.50 (remove all below $0.5)")
    print(f"• Upper threshold: arb_profit <= ${upper_threshold:.6f}")
    print(f"• Original records: {len(df):,}")
    print(f"• High outliers removed: {len(high_outliers):,}")
    print(f"• Low profits removed: {len(low_outliers):,}")
    print(f"• Clean records: {len(clean_df):,}")
    print(f"• Retention rate: {len(clean_df)/len(df)*100:.1f}%")
    
    print(f"\n🎯 FILES CREATED:")
    print(f"• Clean data (profitable): {clean_output_file}")
    print(f"• High outliers removed: {high_outliers_file}")
    print(f"• Low profits removed: {low_profits_file}")
    print(f"• Analysis chart: {plot_output_file}")
    
    print(f"\n✅ COMPLETED! Clean data has removed:")
    print(f"   - Unusual high outliers")
    print(f"   - All profits below $0.5 (too small)")
    print(f"   - Only kept arbitrage transactions with profit >= $0.5")

if __name__ == "__main__":
    main() 