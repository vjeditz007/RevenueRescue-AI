import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

# -----------------------------
# CUSTOMER / PAYMENT FEATURES
# -----------------------------
customer_id = [f"C{1000+i}" for i in np.random.randint(1, 900, N)]

amount = np.round(np.random.lognormal(mean=7.2, sigma=0.75, size=N), 2)
amount = np.clip(amount, 100, 50000)

previous_successes = np.random.poisson(5, N)
previous_failures = np.random.poisson(1.2, N)

customer_age_days = np.random.randint(10, 1500, N)

is_subscription = np.random.binomial(1, 0.42, N)
is_active = np.random.binomial(1, 0.72, N)

failure_reasons = np.random.choice(
    [
        "temporary_failure",
        "expired_method",
        "insufficient_funds",
        "authentication_failure",
        "checkout_abandoned",
        "technical_error"
    ],
    N,
    p=[0.22, 0.18, 0.18, 0.12, 0.15, 0.15]
)

# -----------------------------
# PAYMENT FAILURE GENERATION
# -----------------------------
failed_prob = (
    0.14
    + 0.12 * (previous_failures >= 3)
    + 0.08 * (customer_age_days < 60)
    + 0.10 * (is_active == 0)
    + 0.05 * (failure_reasons == "technical_error")
    + 0.04 * (failure_reasons == "insufficient_funds")
)

failed_prob = np.clip(failed_prob, 0.05, 0.70)

failed = np.random.binomial(1, failed_prob)

# -----------------------------
# LEARNABLE RECOVERY SIGNAL
# -----------------------------
# Start with a base probability
recovery_prob = np.full(N, 0.45)

# Failure reason has strong influence
recovery_prob += 0.22 * (
    failure_reasons == "temporary_failure"
)

recovery_prob += 0.18 * (
    failure_reasons == "technical_error"
)

recovery_prob += 0.15 * (
    failure_reasons == "expired_method"
)

recovery_prob += 0.12 * (
    failure_reasons == "checkout_abandoned"
)

recovery_prob -= 0.20 * (
    failure_reasons == "insufficient_funds"
)

recovery_prob -= 0.16 * (
    failure_reasons == "authentication_failure"
)

# Customer history
recovery_prob += 0.12 * (
    previous_successes >= 5
)

recovery_prob += 0.08 * (
    previous_successes >= 8
)

recovery_prob -= 0.12 * (
    previous_failures >= 4
)

recovery_prob -= 0.08 * (
    previous_failures >= 6
)

# Customer activity
recovery_prob += 0.10 * is_active

# Subscription customers are slightly easier to recover
recovery_prob += 0.05 * is_subscription

# Older customers tend to be more recoverable
recovery_prob += 0.08 * (
    customer_age_days >= 365
)

# Very new customers are slightly harder
recovery_prob -= 0.08 * (
    customer_age_days < 60
)

# Amount effect
recovery_prob -= 0.05 * (
    amount > 10000
)

# Small realistic noise
recovery_prob += np.random.normal(0, 0.035, N)

recovery_prob = np.clip(
    recovery_prob,
    0.03,
    0.97
)

# -----------------------------
# RECOVERY OUTCOME
# -----------------------------
recovered = np.where(
    failed == 1,
    np.random.binomial(1, recovery_prob),
    1
)

# -----------------------------
# DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "transaction_id": [
        f"T{100000+i}" for i in range(N)
    ],
    "customer_id": customer_id,
    "amount": amount,
    "previous_successes": previous_successes,
    "previous_failures": previous_failures,
    "customer_age_days": customer_age_days,
    "is_subscription": is_subscription,
    "is_active": is_active,
    "failure_reason": failure_reasons,
    "failed": failed,
    "recovered": recovered
})

# -----------------------------
# SAVE DATA
# -----------------------------
df.to_csv(
    "data.csv",
    index=False
)

print("=" * 55)
print("RevenueRescue AI Dataset")
print("=" * 55)
print(f"Total payments       : {len(df):,}")
print(f"Failed payments      : {df['failed'].sum():,}")

failed_df = df[df["failed"] == 1]

print(
    f"Recovered failures   : "
    f"{failed_df['recovered'].sum():,}"
)

print(
    f"Recovery rate        : "
    f"{failed_df['recovered'].mean()*100:.1f}%"
)

print("=" * 55)
print("Created data.csv successfully.")
print("=" * 55)