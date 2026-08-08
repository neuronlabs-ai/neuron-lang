"""
Zillow iBuying Algorithm — Simplified Recreation
Replicates the structural ML failures that cost Zillow $881 million.

The real Zillow Offers algorithm:
  - Used historical home sales data to predict future prices
  - Automatically generated purchase offers based on the Zestimate
  - Deployed $3.8B in capital with insufficient model governance

This recreation demonstrates the EXACT types of bugs that caused the loss.
Every bug below is something PyCheck catches — and Python doesn't.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ═══════════════════════════════════════════════════════
#  STEP 1: Load housing market data
# ═══════════════════════════════════════════════════════

def load_market_data():
    """Simulates loading Zillow's historical MLS data."""
    np.random.seed(42)
    n = 10000
    dates = pd.date_range('2018-01-01', periods=n, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'sqft': np.random.normal(2000, 500, n),
        'bedrooms': np.random.choice([2, 3, 4, 5], n),
        'bathrooms': np.random.choice([1, 2, 3], n),
        'lot_size': np.random.normal(8000, 2000, n),
        'year_built': np.random.randint(1960, 2022, n),
        'zip_code': np.random.choice(['85001', '85003', '85004', '85006'], n),
        'market_index': np.cumsum(np.random.normal(0.001, 0.02, n)),  # trending market
        'days_on_market': np.random.exponential(30, n),
    })
    
    # True price = function of features + market trend + noise
    df['sale_price'] = (
        df['sqft'] * 150 +
        df['bedrooms'] * 25000 +
        df['bathrooms'] * 15000 +
        df['lot_size'] * 10 +
        df['market_index'] * 100000 +
        np.random.normal(0, 20000, n)
    )
    
    return df.set_index('date')


# ═══════════════════════════════════════════════════════
#  STEP 2: Feature Engineering — WHERE THE BUGS LIVE
# ═══════════════════════════════════════════════════════

def engineer_features(df):
    """
    Feature engineering pipeline with temporal leakage.
    These are the EXACT types of bugs that caused Zillow's failure.
    """
    
    # BUG 1: Future price change as a feature
    # This is the cardinal sin — using tomorrow's price to predict today's
    df['price_change_next_day'] = df['sale_price'].shift(-1) - df['sale_price']
    
    # BUG 2: Future moving average (negative shift)
    df['future_avg_price'] = df['sale_price'].shift(-30)
    
    # BUG 3: Rolling statistics on the FULL dataset (before train/test split)
    # Zillow computed market-wide statistics including future sales
    df['rolling_median_price'] = df['sale_price'].rolling(90).median()
    df['rolling_std_price'] = df['sale_price'].rolling(90).std()
    df['expanding_mean'] = df['sale_price'].expanding().mean()
    
    # BUG 4: Backfill missing values with FUTURE data
    df['price_change_next_day'] = df['price_change_next_day'].bfill()
    
    # BUG 5: Centered rolling window (uses future data on both sides)
    df['centered_trend'] = df['sale_price'].rolling(60, center=True).mean()
    
    # BUG 6: Correlation-based feature selection (correlation != causation)
    feature_importance = df.corrwith(df['sale_price'])
    
    # BUG 7: Future difference
    df['price_momentum'] = df['sale_price'].diff(-7)
    
    df = df.dropna()
    return df


# ═══════════════════════════════════════════════════════
#  STEP 3: Model Training — MORE BUGS
# ═══════════════════════════════════════════════════════

def train_zestimate_model(df):
    """Train the price prediction model with data leakage."""
    
    feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'lot_size', 'year_built',
                    'market_index', 'days_on_market', 'rolling_median_price',
                    'rolling_std_price', 'expanding_mean', 'price_change_next_day',
                    'future_avg_price', 'centered_trend', 'price_momentum']
    
    X = df[feature_cols]
    y = df['sale_price']
    
    # BUG 8: StandardScaler fitted on ALL data before split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # BUG 9: Random train_test_split on TIME SERIES data
    # Zillow's data is chronological — random splitting leaks future into training
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # BUG 10: KFold cross-validation on time series
    cv = KFold(n_splits=5, shuffle=True)
    
    # Train the "Zestimate" model
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # BUG 11: Evaluate on training data (overfitting check)
    train_score = model.score(X_train, y_train)
    
    # BUG 12: Point predictions without uncertainty
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    print(f"Zestimate MAE: ${mae:,.0f}")
    print(f"Train R²: {train_score:.4f}")  # Looks amazing because of leakage!
    print(f"Test R²:  {model.score(X_test, y_test):.4f}")
    
    return model, scaler


# ═══════════════════════════════════════════════════════
#  STEP 4: Automated iBuying Decisions — THE $881M MISTAKE
# ═══════════════════════════════════════════════════════

def ibuying_engine(model, df):
    """
    Zillow's automated home purchasing engine.
    Makes buy/sell decisions based on Zestimate predictions.
    """
    purchases = 0
    total_spent = 0
    
    for i in range(len(df) - 1):
        current_price = df['sale_price'].iloc[i]
        # BUG 13: Accessing future price in decision loop
        future_price = df['sale_price'].iloc[i + 1]
        
        # BUG 14: Hardcoded prediction threshold
        prediction = model.predict(df[['sqft']].iloc[[i]])
        if prediction > 0.5:
            purchases += 1
            total_spent += current_price
    
    print(f"Total homes purchased: {purchases}")
    print(f"Total capital deployed: ${total_spent:,.0f}")
    
    # BUG 15: No uncertainty quantification
    # Zillow deployed $3.8B without confidence intervals
    final_predictions = model.predict(df[['sqft']].values)
    
    # BUG 16: Using correlation for pricing decisions
    price_corr = df['sqft'].corr()
    
    return purchases


# ═══════════════════════════════════════════════════════
#  STEP 5: "Post-mortem" analysis with same bugs
# ═══════════════════════════════════════════════════════

def postmortem_analysis(df):
    """After losing $881M, analyze what went wrong — but with the same bugs."""
    
    # BUG 17: Selection on outcome (survivorship bias)
    # Only analyzing profitable deals
    profitable = df[df['profit'] > 0]
    
    # BUG 18: P-hacking in the post-mortem
    if p_value < 0.05:
        print("Market shift was statistically significant")
    
    # BUG 19: Silent exception handling hides data issues
    try:
        results = pd.read_csv("zillow_results.csv")
    except:
        pass


if __name__ == "__main__":
    print("=== Zillow iBuying Algorithm Recreation ===")
    print("=== Demonstrating the bugs that cost $881M ===\n")
    
    df = load_market_data()
    df = engineer_features(df)
    model, scaler = train_zestimate_model(df)
    ibuying_engine(model, df)
