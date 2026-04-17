import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_LEGIT = 950
N_FRAUD = 50
START_DATE = datetime(2024, 1, 1)

MERCHANTS = {
    "grocery":    ["Walmart", "Kroger", "Whole Foods", "Trader Joe's", "Aldi"],
    "restaurant": ["McDonald's", "Starbucks", "Chipotle", "Subway", "Domino's"],
    "gas":        ["Shell", "BP", "ExxonMobil", "Chevron", "Circle K"],
    "retail":     ["Amazon", "Target", "Best Buy", "Home Depot", "Costco"],
    "pharmacy":   ["CVS", "Walgreens", "Rite Aid"],
}

COUNTRIES = ["US"] * 90 + ["CA", "MX", "GB", "DE", "FR", "BR", "CN", "RU", "NG"] * 1

def random_merchant():
    category = random.choice(list(MERCHANTS.keys()))
    name = random.choice(MERCHANTS[category])
    return name, category

def generate_legitimate_transactions(n):
    records = []
    raw_probs = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                 0.02, 0.04, 0.06, 0.07, 0.07, 0.08,
                 0.09, 0.09, 0.07, 0.07, 0.06, 0.05,
                 0.05, 0.04, 0.03, 0.02, 0.01, 0.005]
    total = sum(raw_probs)
    hour_probs = [p / total for p in raw_probs]

    for _ in range(n):
        hour = int(np.random.choice(range(24), p=hour_probs))
        minute = random.randint(0, 59)
        day_offset = random.randint(0, 364)
        timestamp = START_DATE + timedelta(days=day_offset, hours=hour, minutes=minute)

        merchant, category = random_merchant()

        amount_ranges = {
            "grocery":    (15, 200),
            "restaurant": (8, 80),
            "gas":        (20, 120),
            "retail":     (10, 500),
            "pharmacy":   (5, 150),
        }
        lo, hi = amount_ranges[category]
        amount = round(random.uniform(lo, hi), 2)

        records.append({
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "timestamp":      timestamp,
            "amount":         amount,
            "merchant":       merchant,
            "category":       category,
            "country":        random.choice(COUNTRIES),
            "hour":           hour,
            "is_fraud":       0,
            "fraud_type":     None,
        })
    return records

def generate_fraud_transactions(n):
    records = []

    for _ in range(n // 3):
        hour = random.randint(1, 4)
        day_offset = random.randint(0, 364)
        timestamp = START_DATE + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))
        merchant, category = random_merchant()
        records.append({
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "timestamp":      timestamp,
            "amount":         round(random.uniform(200, 1500), 2),
            "merchant":       merchant,
            "category":       category,
            "country":        random.choice(["RU", "NG", "CN", "BR"]),
            "hour":           hour,
            "is_fraud":       1,
            "fraud_type":     "unusual_hour",
        })

    thresholds = [499.99, 999.99, 4999.99]
    for _ in range(n // 3):
        base = random.choice(thresholds)
        amount = round(base - random.uniform(0.01, 5.00), 2)
        hour = random.randint(8, 20)
        day_offset = random.randint(0, 364)
        timestamp = START_DATE + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))
        merchant, category = random_merchant()
        records.append({
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "timestamp":      timestamp,
            "amount":         amount,
            "merchant":       merchant,
            "category":       category,
            "country":        random.choice(COUNTRIES),
            "hour":           hour,
            "is_fraud":       1,
            "fraud_type":     "threshold_avoidance",
        })

    fake_merchants = ["QuickPay LLC", "GlobalTrade X", "FastCash Pro", "ZipMerchant", "NovaShop Int'l"]
    for _ in range(n - 2 * (n // 3)):
        hour = random.randint(6, 23)
        day_offset = random.randint(0, 364)
        timestamp = START_DATE + timedelta(days=day_offset, hours=hour, minutes=random.randint(0, 59))
        records.append({
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "timestamp":      timestamp,
            "amount":         round(random.uniform(300, 2000), 2),
            "merchant":       random.choice(fake_merchants),
            "category":       "unknown",
            "country":        random.choice(["US", "MX", "BR", "NG"]),
            "hour":           hour,
            "is_fraud":       1,
            "fraud_type":     "new_merchant_spike",
        })

    return records

def generate_dataset():
    print("Generating legitimate transactions...")
    legit = generate_legitimate_transactions(N_LEGIT)

    print("Injecting fraud patterns...")
    fraud = generate_fraud_transactions(N_FRAUD)

    df = pd.DataFrame(legit + fraud)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    output_path = "data/raw/transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Total: {len(df)} transactions | Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(10))