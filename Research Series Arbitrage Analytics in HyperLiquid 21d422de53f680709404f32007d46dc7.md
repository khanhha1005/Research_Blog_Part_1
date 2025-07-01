# Research Series : Arbitrage Analytics in HyperLiquid

# Introduction

- **Idea 1:** **A ML model to predict which tokens are likely to have arbitrage opportunities in the next 24 hours** .

**Hyperliquid’s** architecture — as the largest causal, fully on-chain DEX — creates frequent and observable arbitrage opportunities. More than just **a trading edge, arbitrage** is a core **mechanism** that drives **price alignment and market efficiency in modern DEX**s.

# Idea 1: Predictive Model for Token Arbitrage Opportunities

## **Goal**

 Develop a machine learning model that predicts which tokens are most likely to present significant arbitrage opportunities in the next 24 hours. This model will help us rank tokens by an "expected arbitrage profit" metric, defined as **Probability of Arbitrage × Potential Volume** (in other words, the likelihood of an arbitrage event times the likely trade volume if it occurs). Such a model can serve as an early warning system for traders, highlighting where to focus attention each day.

## **Approach**

 We will reframe the original analysis pipeline into a clear machine-learning project structure. The model will likely be a classification or regression model that outputs an arbitrage probability and expected profit for each token.

- **Objective:** Identify tokens likely to have price discrepancies that can be exploited for profit within the next 24 hours, and estimate the expected profit from those arbitrage trades.
- **Arbitrage Labeling Logic:** We need a reliable way to label when an arbitrage opportunity actually occurred for a token, so the model can learn from examples. We will likely define a **threshold-based label**, for example: *Did the token experience a price spread above X% between Hyperliquid and other markets, followed by a quick convergence (suggesting arbitrage trades closed the gap)?* If yes, mark that token (for that day) as having an arbitrage event. We’ll use a combination of HyperCore price vs. global price, and possibly the presence of large inter-chain transfers, to label positive instances. Each token-day could be a data point labeled "Arbitrage occurred" or "No arbitrage".
- **Model Training:**
    - We will likely start with a supervised learning approach. A classification model (like a tree-based model or logistic regression, or even a simple neural network) can predict the probability of arbitrage for each token each day.
    - We might also train a regression model to predict the size of the price spread or profit directly. A reasonable approach is to train a gradient boosting model (e.g. XGBoost or LightGBM) on historical data, as these handle mixed features well and are interpretable. We will take care to avoid leakage (ensuring we only use information that would have been known *before* the arbitrage event to predict it).
- **Inference Logic:** Once the model is trained, each day (or each hour) we can feed in the latest data to get a probability of arbitrage for each token, and an expected profit (probability × estimated volume * price discrepancy). Tokens can then be ranked by this **Expected Arbitrage Profit** metric. The top-ranked tokens are the ones to watch or potentially act on.
    - For example, if the model says Token A has 80% chance of a 5% price discrepancy on ~$10M volume, that yields an expected value suggesting a significant profit opportunity, higher than Token B with 30% chance on $2M volume.
- **Expected Output:** The outcome will be a dashboard and report highlighting the tokens with highest arbitrage potential. An **illustrative output table** might look like:
    
    
    | Token | Probability of Arb (Next 24h) | Predicted Volume Gap | Expected Profit Potential |
    | --- | --- | --- | --- |
    | HYPE | **60%** (High) | $1.2M | $720k (USD) |
    | ETH | 35% (Moderate) | $3.0M | $1.05M |
    | ARB | 20% (Low) | $500k | $100k |
    |  |  |  |  |

In the actual blog post, we will include visualizations such as time series of price spreads for top tokens, and feature importance charts explaining what signals the model is using. By the end of this segment of the series, readers should understand how predictive modeling can identify arbitrage opportunities and how to interpret the model’s output in a practical strategy context.

|  | block_time | time | amount_HYPE | price_HYPE_EVM | price_HYPE_Core | delta_USD | arb_profit | block_number | tx_hash | hash |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2025-05-23 16:16:00+00:00 | 2025-05-23 16:16:00.039442939+00:00 | 6.5 | 34.73713 | 34.702 | 0.035134 | 0.22837 | 4218814 | 0xb45f96a8c4aff4b3451a49d4d1d2ee2cfc87715c5042d2c2c878ef8f74b2573f_2 | 0x526a1505e92ba53242cd04240b2d4e02016300cb8630ee12e8378fd9bfd8d550 |  |  |  |  |  |  |  |
| 3 | 2025-05-23 16:17:00+00:00 | 2025-05-23 16:17:00.062408340+00:00 | 7.1 | 34.73713 | 34.669 | 0.068134 | 0.48375 | 4218844 | 0x7946ee22dfddb4339174e720e55808e8b57152cc9a224a3710c936632c4f630a_13 | 0x0000000000000000000000000000000000000000000000000000000000000000 |  |  |  |  |  |  |  |
| 4 | 2025-05-23 16:17:48+00:00 | 2025-05-23 16:17:48.204487856+00:00 | 21.3 | 34.73713 | 34.6 | 0.137134 | 2.92095 | 4218869 | 0x3715c3ee1db6ac6b798528e742d56c2d17f3917c32abccf8d122ca507532e124_0 | 0x76de5a213caa1a0e600004240b32ba0205470001682795e688fccafbb9f04629 |  |  |  |  |  |  |  |
| 5 | 2025-05-23 16:18:14+00:00 | 2025-05-23 16:18:14.042121876+00:00 | 14 | 34.73713 | 34.564 | 0.173134 | 2.423873 | 4218883 | 0x4270f895565b4f70fe014664410ae15557318955633dd3d17b2493da249c6057_5 | 0x0000000000000000000000000000000000000000000000000000000000000000 |  |  |  |  |  |  |  |
| 6 | 2025-05-23 16:18:34+00:00 | 2025-05-23 16:18:34.025455835+00:00 | 8.852682 | 34.55267 | 34.506 | 0.046671 | 0.413165 | 4218893 | 0x9558c58cb7679cebde7d7580b5b9cd8ce659ea0600d23eb2bd74ce273b153177_0 | 0x0000000000000000000000000000000000000000000000000000000000000000 |  |  |  |  |  |  |  |

---

## 1. Raw (“Primitive”) Fields

| Field | Description |
| --- | --- |
| `block_time` | Block timestamp (UTC) |
| `amount_HYPE` | Amount of HYPE traded |
| `price_HYPE_EVM` | HYPE price on the EVM DEX (USD) |
| `price_HYPE_Core` | HYPE price on Hyperliquid Core (USD) |
| `delta_USD` | Price spread = `price_HYPE_EVM – price_HYPE_Core` |
| `arb_profit` | Actual arbitrage profit (USD) for that trade |

> The fields block_number, tx_hash, and hash serve only as identifiers and are not used as features.
> 

---

## 2. Engineered (“Technical”) Features

*All aggregated over a fixed “feature window” before prediction time t₀ (e.g. the 1 hour prior to t₀).*

| Feature Name | Definition / Notes |
| --- | --- |
| **Price Spread** |  |
| - `spread_mean_1h` | Mean of `delta_USD` over the 1 hour window |
| - `spread_std_1h` | Standard deviation of `delta_USD` over 1 hour |
| - `spread_max_1h` | Maximum `delta_USD` in the past hour |
| - `spread_min_1h` | Minimum `delta_USD` in the past hour |
| **Volume & Activity** |  |
| - `vol_sum_1h` | Total `amount_HYPE` traded in 1 hour |
| - `vol_count_1h` | Number of trades (rows) in the past hour |
| - `vol_large_trades_1h` | Number of trades with `amount_HYPE` > 90th percentile in the past hour |
| **Momentum & Volatility** |  |
| - `ret_5m` | 5-minute momentum: `(price_end – price_start) / price_start` in 5 minutes |
| - `vol_realized_5m` | Std dev of 1-minute returns over the past 5 minutes |
| **Temporal** |  |
| - `hour_of_day` | Hour of day (0–23) |
| - `day_of_week` | Day of week (0 = Sunday … 6 = Saturday) |

---

## 3. Model Input & Label

- **Sample**: one token at time t₀ (e.g. each hourly snapshot).
- **Input features**: the engineered features computed up to t₀, for example:
    
    ```json
    {
      "spread_mean_1h": …,
      "spread_std_1h": …,
      "spread_max_1h": …,
      "spread_min_1h": …,
      "vol_sum_1h": …,
      "vol_count_1h": …,
      "vol_large_trades_1h": …,
      "ret_5m": …,
      "vol_realized_5m": …,
      "hour_of_day": …,
      "day_of_week": …
    }
    
    ```
    
- **Label**:
    - **Classification**: 1 if an arbitrage event (≥ X% profit) occurs in the 24 hours following t₀, otherwise 0.
    - (Or you can build a regression target = actual profit.)

---