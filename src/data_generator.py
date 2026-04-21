"""
SynthAudit — Synthetic Financial Transaction Generator
-------------------------------------------------------
Generates a realistic dataset of financial transactions with injected
fraud patterns for training and evaluating ML-based fraud detection models.

Fraud patterns implemented:
  1. unusual_hour      — Transactions at 1–4 AM from high-risk countries
  2. threshold_avoidance — Amounts just below common detection thresholds
  3. new_merchant_spike — Large amounts from unknown merchant entities
  4. velocity_burst     — Multiple transactions in rapid succession (same card)
  5. geo_mismatch       — High-value transactions from unexpected countries

Scale: 50,000 transactions (~2.5% fraud rate ≈ 1,250 fraud records)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_LEGIT  = 48_750
N_FRAUD  = 1_250
START_DATE = datetime(2023, 1, 1)

MERCHANTS = {
    "grocery":    ["Walmart", "Kroger", "Whole Foods", "Trader Joe's", "Aldi",
                   "Safeway", "Publix", "H-E-B"],
    "restaurant": ["McDonald's", "Starbucks", "Chipotle", "Subway", "Domino's",
                   "Chick-fil-A", "Panera", "Taco Bell", "Olive Garden"],
    "gas":        ["Shell", "BP", "ExxonMobil", "Chevron", "Circle K",
                   "Sunoco", "Valero"],
    "retail":     ["Amazon", "Target", "Best Buy", "Home Depot", "Costco",
                   "Walmart", "Macy's", "Nordstrom", "IKEA"],
    "pharmacy":   ["CVS", "Walgreens", "Rite Aid", "Duane Reade"],
    "travel":     ["Delta Airlines", "United Airlines", "Marriott", "Hilton",
                   "Airbnb", "Booking.com", "Hertz", "Enterprise"],
    "streaming":  ["Netflix", "Spotify", "Disney+", "Hulu", "Apple TV+"],
    "utilities":  ["Verizon", "AT&T", "Comcast", "Duke Energy", "Con Edison"],
}

# Realistic domestic / international split
DOMESTIC_COUNTRIES  = ["US"] * 88
FOREIGN_COUNTRIES   = ["CA", "MX", "GB", "DE", "FR", "AU", "JP", "IN",
                       "BR", "CN", "RU", "NG", "UA", "PK", "KE"] * 1
ALL_COUNTRIES = DOMESTIC_COUNTRIES + FOREIGN_COUNTRIES

HIGH_RISK_COUNTRIES = ["RU", "NG", "CN", "BR", "UA", "PK", "KE"]

AMOUNT_RANGES = {
    "grocery":    (10,  300),
    "restaurant": (5,   120),
    "gas":        (15,  150),
    "retail":     (10,  800),
    "pharmacy":   (5,   200),
    "travel":     (80, 3000),
    "streaming":  (8,   25),
    "utilities":  (30,  400),
}

FAKE_MERCHANTS = [
    "QuickPay LLC", "GlobalTrade X", "FastCash Pro", "ZipMerchant",
    "NovaShop Int'l", "SpeedFund Co", "FlexPay Solutions", "RapidMerch Ltd",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def random_merchant():
    category = random.choice(list(MERCHANTS.keys()))
    name = random.choice(MERCHANTS[category])
    return name, category


def random_timestamp(hour_probs=None, days_range=730):
    if hour_probs is None:
        # Realistic daytime distribution
        raw = [0.005]*6 + [0.02, 0.04, 0.07, 0.08, 0.09, 0.09,
                           0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
                           0.04, 0.03, 0.02, 0.015, 0.01, 0.005]
        hour_probs = [p / sum(raw) for p in raw]
    hour   = int(np.random.choice(range(24), p=hour_probs))
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    day    = random.randint(0, days_range - 1)
    return START_DATE + timedelta(days=day, hours=hour, minutes=minute,
                                  seconds=second), hour


# ── Legitimate transactions ──────────────────────────────────────────────────
def generate_legitimate_transactions(n):
    records = []
    raw = [0.005]*6 + [0.02, 0.04, 0.07, 0.08, 0.09, 0.09,
                       0.10, 0.09, 0.08, 0.07, 0.06, 0.05,
                       0.04, 0.03, 0.02, 0.015, 0.01, 0.005]
    hour_probs = [p / sum(raw) for p in raw]

    for _ in range(n):
        ts, hour     = random_timestamp(hour_probs)
        merchant, cat = random_merchant()
        lo, hi       = AMOUNT_RANGES[cat]
        amount       = round(random.uniform(lo, hi), 2)
        country      = random.choice(ALL_COUNTRIES)

        records.append({
            "transaction_id": f"TXN{random.randint(1_000_000, 9_999_999)}",
            "timestamp":      ts,
            "amount":         amount,
            "merchant":       merchant,
            "category":       cat,
            "country":        country,
            "hour":           hour,
            "day_of_week":    ts.weekday(),          # 0=Mon, 6=Sun
            "is_weekend":     int(ts.weekday() >= 5),
            "is_fraud":       0,
            "fraud_type":     None,
        })
    return records


# ── Fraud transactions ───────────────────────────────────────────────────────
def generate_fraud_transactions(n):
    records = []
    counts  = {
        "unusual_hour":       n // 5,
        "threshold_avoidance": n // 5,
        "new_merchant_spike": n // 5,
        "velocity_burst":     n // 5,
        "geo_mismatch":       n - 4 * (n // 5),   # absorbs rounding
    }

    # 1 — Unusual hour (1–4 AM, high-risk country, elevated amount)
    for _ in range(counts["unusual_hour"]):
        hour   = random.randint(1, 4)
        day    = random.randint(0, 729)
        ts     = START_DATE + timedelta(days=day, hours=hour,
                                        minutes=random.randint(0, 59))
        merchant, cat = random_merchant()
        records.append({
            "transaction_id": f"TXN{random.randint(1_000_000, 9_999_999)}",
            "timestamp":      ts,
            "amount":         round(random.uniform(300, 2500), 2),
            "merchant":       merchant,
            "category":       cat,
            "country":        random.choice(HIGH_RISK_COUNTRIES),
            "hour":           hour,
            "day_of_week":    ts.weekday(),
            "is_weekend":     int(ts.weekday() >= 5),
            "is_fraud":       1,
            "fraud_type":     "unusual_hour",
        })

    # 2 — Threshold avoidance (amount just below $500 / $1k / $5k)
    thresholds = [499.99, 999.99, 4_999.99]
    for _ in range(counts["threshold_avoidance"]):
        base   = random.choice(thresholds)
        amount = round(base - random.uniform(0.50, 10.00), 2)
        ts, hour = random_timestamp()
        merchant, cat = random_merchant()
        records.append({
            "transaction_id": f"TXN{random.randint(1_000_000, 9_999_999)}",
            "timestamp":      ts,
            "amount":         amount,
            "merchant":       merchant,
            "category":       cat,
            "country":        random.choice(ALL_COUNTRIES),
            "hour":           hour,
            "day_of_week":    ts.weekday(),
            "is_weekend":     int(ts.weekday() >= 5),
            "is_fraud":       1,
            "fraud_type":     "threshold_avoidance",
        })

    # 3 — New merchant spike (unknown entity, large amount)
    for _ in range(counts["new_merchant_spike"]):
        ts, hour = random_timestamp()
        records.append({
            "transaction_id": f"TXN{random.randint(1_000_000, 9_999_999)}",
            "timestamp":      ts,
            "amount":         round(random.uniform(400, 3000), 2),
            "merchant":       random.choice(FAKE_MERCHANTS),
            "category":       "unknown",
            "country":        random.choice(["US", "MX", "BR", "NG"]),
            "hour":           hour,
            "day_of_week":    ts.weekday(),
            "is_weekend":     int(ts.weekday() >= 5),
            "is_fraud":       1,
            "fraud_type":     "new_merchant_spike",
        })

    # 4 — Velocity burst (cluster of transactions within 15 minutes)
    for _ in range(counts["velocity_burst"]):
        base_ts, _ = random_timestamp()
        for i in range(random.randint(3, 6)):
            ts   = base_ts + timedelta(minutes=random.randint(0, 15))
            hour = ts.hour
            merchant, cat = random_merchant()
            records.append({
                "transaction_id": f"TXN{random.randint(1_000_000, 9_999_999)}",
                "timestamp":      ts,
                "amount":         round(random.uniform(50, 800), 2),
                "merchant":       merchant,
                "category":       cat,
                "country":        random.choice(ALL_COUNTRIES),
                "hour":           hour,
                "day_of_week":    ts.weekday(),
                "is_weekend":     int(ts.weekday() >= 5),
                "is_fraud":       1,
                "fraud_type":     "velocity_burst",
            })
            if len(records) >= (counts["unusual_hour"] +
                                counts["threshold_avoidance"] +
                                counts["new_merchant_spike"] +
                                counts["velocity_burst"]):
                break
        else:
            continue
        break

    # 5 — Geo mismatch (high-value txn from unexpected high-risk country)
    for _ in range(counts["geo_mismatch"]):
        ts, hour = random_timestamp()
        merchant, cat = random_merchant()
        records.append({
            "transaction_id": f"TXN{random.randint(1_000_000, 9_999_999)}",
            "timestamp":      ts,
            "amount":         round(random.uniform(500, 4000), 2),
            "merchant":       merchant,
            "category":       cat,
            "country":        random.choice(HIGH_RISK_COUNTRIES),
            "hour":           hour,
            "day_of_week":    ts.weekday(),
            "is_weekend":     int(ts.weekday() >= 5),
            "is_fraud":       1,
            "fraud_type":     "geo_mismatch",
        })

    return records


# ── Main ─────────────────────────────────────────────────────────────────────
def generate_dataset(output_path="data/raw/transactions.csv"):
    print(f"Generating {N_LEGIT:,} legitimate transactions...")
    legit = generate_legitimate_transactions(N_LEGIT)

    print(f"Injecting {N_FRAUD:,} fraud transactions across 5 patterns...")
    fraud = generate_fraud_transactions(N_FRAUD)

    df = pd.DataFrame(legit + fraud)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_csv(output_path, index=False)

    total  = len(df)
    n_fraud = df["is_fraud"].sum()
    print(f"\n✓ Dataset saved → {output_path}")
    print(f"  Total rows : {total:,}")
    print(f"  Fraud rows : {n_fraud:,}  ({n_fraud/total*100:.2f}%)")
    print(f"  Legit rows : {total - n_fraud:,}")
    print(f"\n  Fraud breakdown:")
    for ftype, grp in df[df["is_fraud"] == 1].groupby("fraud_type"):
        print(f"    {ftype:<25} {len(grp):>5,} rows")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print("\nSample rows:")
    print(df.head(10).to_string())
