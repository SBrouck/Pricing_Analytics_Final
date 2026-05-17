"""
Synthetic transaction-level dataset for campus coffee shop promotion MAB experiments.
Colab-friendly: run as script or import functions.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMOTION_ARMS = [
    "no_discount",
    "discount_5",
    "discount_10",
    "discount_15",
    "bundle",
]

DISCOUNT_RATE_MAP = {
    "no_discount": 0.0,
    "discount_5": 0.05,
    "discount_10": 0.10,
    "discount_15": 0.15,
    "bundle": 0.12,  # effective bundled discount vs list price
}

BUNDLE_FIXED_PRICE = 8.50
N_TRANSACTIONS = 10_000
RANDOM_SEED = 42

CUSTOMER_TYPE_PROBS = {"new": 0.30, "returning": 0.45, "loyalty_member": 0.25}
TIME_OF_DAY_PROBS = {"morning": 0.32, "afternoon": 0.42, "evening": 0.26}
WEATHER_PROBS = {"sunny": 0.44, "cloudy": 0.36, "rainy": 0.20}
QUEUE_PROBS = {"short": 0.48, "medium": 0.37, "long": 0.15}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Customer pool
# ---------------------------------------------------------------------------

def generate_customers(
    n_transactions: int,
    rng: np.random.Generator,
    n_unique_customers: int = 2_800,
) -> pd.DataFrame:
    """Create customer master records and map each transaction to a customer."""
    types = rng.choice(
        list(CUSTOMER_TYPE_PROBS),
        size=n_unique_customers,
        p=list(CUSTOMER_TYPE_PROBS.values()),
    )
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_unique_customers + 1)]

    # Visit weights: loyalty members return more often
    type_weight = {"new": 0.7, "returning": 1.0, "loyalty_member": 2.2}
    weights = np.array([type_weight[t] for t in types])
    weights /= weights.sum()

    chosen_idx = rng.choice(n_unique_customers, size=n_transactions, p=weights)
    txn_customer_ids = [customer_ids[i] for i in chosen_idx]
    txn_types = [types[i] for i in chosen_idx]

    repeat_visits = np.zeros(n_transactions, dtype=int)
    for cust_idx in np.unique(chosen_idx):
        mask = chosen_idx == cust_idx
        n_visits = mask.sum()
        ctype = types[cust_idx]
        if ctype == "new":
            lam = 1.1
        elif ctype == "returning":
            lam = 2.4
        else:
            lam = 4.0
        repeat_visits[mask] = rng.poisson(lam=lam, size=n_visits)

    return pd.DataFrame(
        {
            "customer_id": txn_customer_ids,
            "customer_type": txn_types,
            "loyalty_membership": (np.array(txn_types) == "loyalty_member").astype(int),
            "repeat_visit_this_week": repeat_visits,
        }
    )


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def _sample_order_method(customer_type: str, rng: np.random.Generator) -> str:
    if customer_type == "loyalty_member":
        p = [0.52, 0.18, 0.30]  # app, counter, kiosk
    elif customer_type == "returning":
        p = [0.28, 0.42, 0.30]
    else:
        p = [0.18, 0.48, 0.34]
    return rng.choice(["app", "counter", "kiosk"], p=p)


def _sample_dine_mode(order_method: str, weather: str, rng: np.random.Generator) -> str:
    if order_method == "app":
        p = [0.12, 0.22, 0.66]  # dine_in, takeout, mobile_pickup
    else:
        p = [0.42, 0.48, 0.10]
    if weather == "rainy":
        p = np.array(p) * np.array([0.75, 1.05, 1.15])
        p /= p.sum()
    return rng.choice(["dine_in", "takeout", "mobile_pickup"], p=p)


def assign_context(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(customers)
    time_of_day = rng.choice(
        list(TIME_OF_DAY_PROBS), size=n, p=list(TIME_OF_DAY_PROBS.values())
    )
    day_of_week = rng.choice(DAYS, size=n, p=[0.16, 0.15, 0.15, 0.15, 0.17, 0.11, 0.11])
    is_finals_week = rng.binomial(1, 0.14, size=n)
    weather = rng.choice(
        list(WEATHER_PROBS), size=n, p=list(WEATHER_PROBS.values())
    )
    queue_length = rng.choice(
        list(QUEUE_PROBS), size=n, p=list(QUEUE_PROBS.values())
    )

    # Afternoon queues slightly longer (off-peak staffing)
    afternoon_mask = time_of_day == "afternoon"
    if afternoon_mask.any():
        requeue = rng.choice(
            ["short", "medium", "long"],
            size=afternoon_mask.sum(),
            p=[0.38, 0.40, 0.22],
        )
        queue_length = queue_length.copy()
        queue_length[afternoon_mask] = requeue

    order_method = [
        _sample_order_method(ct, rng) for ct in customers["customer_type"]
    ]
    dine_in_takeout = [
        _sample_dine_mode(om, w, rng)
        for om, w in zip(order_method, weather)
    ]

    return pd.DataFrame(
        {
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "is_finals_week": is_finals_week,
            "weather": weather,
            "queue_length": queue_length,
            "order_method": order_method,
            "dine_in_takeout": dine_in_takeout,
        }
    )


# ---------------------------------------------------------------------------
# Promotion assignment (pluggable policy)
# ---------------------------------------------------------------------------

def assign_promotions(
    n: int,
    rng: np.random.Generator,
    policy: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Assign promotion arms. Default: uniform random (static experiment).
    Pass policy(n, rng) -> array of arm names for adaptive bandits later.
    """
    if policy is None:
        arms = rng.choice(PROMOTION_ARMS, size=n)
    else:
        arms = policy(n, rng)

    return pd.DataFrame(
        {
            "promotion_shown": arms,
            "discount_rate": [DISCOUNT_RATE_MAP[a] for a in arms],
        }
    )


# ---------------------------------------------------------------------------
# Conversion (logistic latent utility)
# ---------------------------------------------------------------------------

def _compute_conversion_logit(df: pd.DataFrame) -> np.ndarray:
    eta = np.full(len(df), -0.78)  # ~31% baseline at reference context

    # Customer type
    eta += np.where(df["customer_type"] == "new", -0.48, 0.0)
    eta += np.where(df["customer_type"] == "loyalty_member", 0.58, 0.0)

    # Time of day
    eta += np.where(df["time_of_day"] == "morning", 0.38, 0.0)
    eta += np.where(df["time_of_day"] == "afternoon", -0.28, 0.0)
    eta += np.where(df["time_of_day"] == "evening", 0.06, 0.0)

    # Day of week (weekend bump)
    eta += np.where(df["day_of_week"].isin(["Fri", "Sat"]), 0.12, 0.0)
    eta += np.where(df["day_of_week"] == "Sun", -0.08, 0.0)

    # Finals, weather, queue
    eta += df["is_finals_week"] * 0.22
    eta += np.where(df["weather"] == "rainy", 0.10, 0.0)
    eta += np.where(df["weather"] == "cloudy", 0.03, 0.0)
    eta += np.where(df["queue_length"] == "medium", -0.14, 0.0)
    eta += np.where(df["queue_length"] == "long", -0.42, 0.0)

    # Order method & dine mode
    eta += np.where(df["order_method"] == "app", 0.20, 0.0)
    eta += np.where(df["dine_in_takeout"] == "dine_in", 0.05, 0.0)
    eta += np.where(
        (df["weather"] == "rainy") & (df["dine_in_takeout"] == "dine_in"), -0.14, 0.0
    )

    # Repeat visits (habit)
    eta += np.clip(df["repeat_visit_this_week"] * 0.04, 0, 0.28)

    promo = df["promotion_shown"]
    arm_lift = {
        "no_discount": 0.0,
        "discount_5": 0.14,
        "discount_10": 0.33,
        "discount_15": 0.40,
        "bundle": 0.22,
    }
    for arm, lift in arm_lift.items():
        eta += np.where(promo == arm, lift, 0.0)

    # New customers more price-sensitive
    new_mask = df["customer_type"] == "new"
    for arm in ["discount_5", "discount_10", "discount_15"]:
        eta += np.where(new_mask & (promo == arm), 0.10, 0.0)

    # Discounts stronger in afternoon off-peak
    afternoon = df["time_of_day"] == "afternoon"
    for arm in ["discount_5", "discount_10", "discount_15"]:
        eta += np.where(afternoon & (promo == arm), 0.12, 0.0)

    # Bundle interactions
    eta += np.where(
        (promo == "bundle") & (df["customer_type"] == "loyalty_member"), 0.28, 0.0
    )
    eta += np.where(
        (promo == "bundle") & (df["queue_length"] == "long"), -0.62, 0.0
    )
    eta += np.where(
        (promo == "bundle") & (df["queue_length"] == "medium"), -0.22, 0.0
    )

    # Finals week: bundle and discount_10 more attractive
    eta += np.where(
        (df["is_finals_week"] == 1) & (promo == "bundle"), 0.18, 0.0
    )
    eta += np.where(
        (df["is_finals_week"] == 1) & (promo == "discount_10"), 0.14, 0.0
    )
    eta += df["is_finals_week"] * np.where(promo == "discount_15", 0.08, 0.0)

    # App users respond better to promotions
    for arm in ["discount_10", "discount_15", "bundle"]:
        eta += np.where((df["order_method"] == "app") & (promo == arm), 0.08, 0.0)

    # Loyalty less sensitive to deep discounts (smaller incremental lift for 15%)
    eta += np.where(
        (df["customer_type"] == "loyalty_member") & (promo == "discount_15"),
        -0.10,
        0.0,
    )

    return eta


def simulate_conversion(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    eta = _compute_conversion_logit(df)
    prob = 1 / (1 + np.exp(-eta))
    conversion = rng.binomial(1, prob)
    df = df.copy()
    df["conversion"] = conversion
    return df


# ---------------------------------------------------------------------------
# Basket size & economics
# ---------------------------------------------------------------------------

def _sample_item_prices(n_items: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(4.00, 7.50, size=n_items)


def _sample_basket_size(row: pd.Series, rng: np.random.Generator) -> int:
    if row["promotion_shown"] == "bundle":
        p = [0.15, 0.55, 0.30]  # 1, 2, 3 — bundle usually 2+
        if row["customer_type"] == "loyalty_member":
            p = [0.08, 0.50, 0.42]
        if row["is_finals_week"] == 1:
            p = [0.05, 0.45, 0.50]
        if row["queue_length"] == "long":
            p = [0.35, 0.45, 0.20]
        sizes = [1, 2, 3]
    else:
        p = [0.52, 0.35, 0.13]
        if row["customer_type"] == "loyalty_member":
            p = [0.40, 0.40, 0.20]
        if row["is_finals_week"] == 1:
            p = [0.38, 0.40, 0.22]
        if row["queue_length"] == "long":
            p = [0.62, 0.30, 0.08]
        sizes = [1, 2, 3]

    return int(rng.choice(sizes, p=p))


def _compute_cost(
    price_charged: float,
    basket_size: int,
    promo: str,
    rng: np.random.Generator,
) -> float:
    if promo == "bundle":
        margin_rate = rng.uniform(0.38, 0.48)  # pastry + labor; tighter margin
    elif promo == "discount_15":
        margin_rate = rng.uniform(0.33, 0.44)
    elif promo == "discount_10":
        margin_rate = rng.uniform(0.40, 0.52)
    elif promo == "discount_5":
        margin_rate = rng.uniform(0.38, 0.50)
    else:
        margin_rate = rng.uniform(0.42, 0.55)
    margin_rate -= 0.02 * max(0, basket_size - 1)
    margin_rate = np.clip(margin_rate, 0.30, 0.58)
    cost = price_charged * (1 - margin_rate)
    if rng.random() < 0.02:
        cost = price_charged * rng.uniform(0.88, 0.98)
    return round(float(cost), 2)


def simulate_basket_and_profit(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    basket_size = np.zeros(n, dtype=int)
    base_price = np.zeros(n)
    price_charged = np.zeros(n)
    cost = np.zeros(n)

    converters = df["conversion"] == 1
    for idx in np.where(converters)[0]:
        row = df.iloc[idx]
        bs = _sample_basket_size(row, rng)
        if row["promotion_shown"] == "bundle":
            bs = max(bs, 2)
        basket_size[idx] = bs

        item_prices = _sample_item_prices(bs, rng)
        list_total = float(item_prices.sum())
        base_price[idx] = round(list_total, 2)

        promo = row["promotion_shown"]
        if promo == "bundle":
            charged = BUNDLE_FIXED_PRICE
            if bs >= 3:
                charged += rng.uniform(0.75, 2.00)  # add-on with bundle
        elif promo == "no_discount":
            charged = list_total
        else:
            dr = DISCOUNT_RATE_MAP[promo]
            charged = list_total * (1 - dr)

        charged = round(charged, 2)
        price_charged[idx] = charged
        cost[idx] = _compute_cost(charged, bs, promo=promo, rng=rng)

    df["basket_size"] = basket_size
    df["base_price"] = np.round(base_price, 2)
    df["price_charged"] = np.round(price_charged, 2)
    df["cost"] = np.round(cost, 2)
    df["contribution_profit"] = np.where(
        df["conversion"] == 1,
        np.round(df["price_charged"] - df["cost"], 2),
        0.0,
    )
    return df


def _simulate_response_time(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Seconds from offer shown to decision; lower for app / loyalty / converters."""
    base_mu = {"app": 2.8, "counter": 4.6, "kiosk": 4.0}
    mu = df["order_method"].map(base_mu).to_numpy()
    mu -= df["loyalty_membership"] * 0.35
    mu -= df["conversion"] * 0.45
    mu += np.where(df["queue_length"] == "long", 0.9, 0.0)
    mu += np.where(df["promotion_shown"] == "bundle", 0.25, 0.0)
    sigma = 0.55
    times = rng.lognormal(mean=np.log(np.clip(mu, 0.5, None)), sigma=sigma)
    return np.clip(times, 0.4, 45.0).round(1)


# ---------------------------------------------------------------------------
# Validation & summaries
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> dict:
    checks = {}
    checks["no_missing"] = not df.isnull().any().any()
    checks["conversion_binary"] = set(df["conversion"].unique()).issubset({0, 1})
    zero_conv = df["conversion"] == 0
    for col in ["basket_size", "price_charged", "cost", "contribution_profit"]:
        checks[f"{col}_zero_if_no_conv"] = (df.loc[zero_conv, col] == 0).all()
    conv = df["conversion"] == 1
    checks["profit_nonneg_majority"] = (df.loc[conv, "contribution_profit"] >= 0).mean() > 0.9
    checks["arms_balanced"] = df["promotion_shown"].value_counts().min() > 1500
    checks["conv_rate_range"] = (df["conversion"].mean() >= 0.15) and (
        df.groupby("promotion_shown")["conversion"].mean().max() <= 0.70
    )
    checks["profit_varies_by_arm"] = (
        df.groupby("promotion_shown")["contribution_profit"].mean().std() > 0.08
    )
    return checks


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    conv = df[df["conversion"] == 1]
    summary = (
        df.groupby("promotion_shown")
        .agg(
            n=("transaction_id", "count"),
            mean_conversion=("conversion", "mean"),
            mean_contribution_profit=("contribution_profit", "mean"),
            total_contribution_profit=("contribution_profit", "sum"),
        )
        .reset_index()
    )
    conv_summary = (
        conv.groupby("promotion_shown")
        .agg(
            mean_basket_size_among_converters=("basket_size", "mean"),
            mean_price_charged_among_converters=("price_charged", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(conv_summary, on="promotion_shown", how="left")
    summary = summary[
        [
            "promotion_shown",
            "n",
            "mean_conversion",
            "mean_basket_size_among_converters",
            "mean_price_charged_among_converters",
            "mean_contribution_profit",
            "total_contribution_profit",
        ]
    ]
    return summary.sort_values("mean_contribution_profit", ascending=False)


def build_data_dictionary() -> pd.DataFrame:
    rows = [
        ("transaction_id", "string", "Unique transaction identifier", "TXN_00001"),
        ("round_number", "int", "Experiment round 1..N", "1"),
        ("customer_id", "string", "Customer identifier (repeat visits allowed)", "CUST_00042"),
        ("promotion_shown", "categorical", "MAB arm shown", "discount_10"),
        ("discount_rate", "float", "Nominal discount rate (bundle uses effective rate)", "0.10"),
        ("conversion", "binary", "1=purchase, 0=no purchase", "1"),
        ("basket_size", "int", "Items purchased; 0 if no conversion", "2"),
        ("base_price", "float", "Pre-discount list basket value", "11.25"),
        ("price_charged", "float", "Revenue after promotion; 0 if no conversion", "10.12"),
        ("cost", "float", "COGS; 0 if no conversion", "5.40"),
        ("contribution_profit", "float", "MAB reward: price_charged - cost; 0 if no conversion", "4.72"),
        ("order_method", "categorical", "app, counter, kiosk", "app"),
        ("dine_in_takeout", "categorical", "dine_in, takeout, mobile_pickup", "mobile_pickup"),
        ("time_of_day", "categorical", "morning, afternoon, evening", "afternoon"),
        ("day_of_week", "categorical", "Mon-Sun", "Wed"),
        ("is_finals_week", "binary", "1 during finals period", "0"),
        ("customer_type", "categorical", "new, returning, loyalty_member", "returning"),
        ("loyalty_membership", "binary", "1 if loyalty member", "0"),
        ("weather", "categorical", "sunny, cloudy, rainy", "rainy"),
        ("queue_length", "categorical", "short, medium, long", "medium"),
        ("repeat_visit_this_week", "int", "Visits by customer in current week", "3"),
        ("promotion_response_time", "float", "Seconds from offer to decision", "3.2"),
    ]
    return pd.DataFrame(rows, columns=["column", "dtype", "description", "example"])


def _print_ground_truth(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\n=== Ground truth: promotion arm performance ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = summary.iloc[0]["promotion_shown"]
    print(f"\nBest arm by mean contribution_profit: {best}")


def _make_plots(df: pd.DataFrame, output_dir: str = ".") -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(8, 5))
    order = (
        df.groupby("promotion_shown")["conversion"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.barplot(
        data=df, x="promotion_shown", y="conversion", order=order, ax=ax, errorbar=("ci", 95)
    )
    ax.set_title("Conversion rate by promotion arm")
    ax.set_xlabel("Promotion shown")
    ax.set_ylabel("Conversion rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conversion_by_arm.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    profit_order = (
        df.groupby("promotion_shown")["contribution_profit"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    sns.barplot(
        data=df,
        x="promotion_shown",
        y="contribution_profit",
        order=profit_order,
        ax=ax,
        errorbar=("ci", 95),
    )
    ax.set_title("Mean contribution profit by promotion arm (incl. zeros)")
    ax.set_xlabel("Promotion shown")
    ax.set_ylabel("Mean contribution profit ($)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_by_arm.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ctx = (
        df.groupby(["time_of_day", "promotion_shown"])["conversion"]
        .mean()
        .reset_index()
    )
    sns.barplot(data=ctx, x="time_of_day", y="conversion", hue="promotion_shown", ax=ax)
    ax.set_title("Conversion by time of day and promotion")
    ax.set_ylabel("Conversion rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "conversion_by_context.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Assembly & I/O
# ---------------------------------------------------------------------------

def generate_dataset(
    n: int = N_TRANSACTIONS,
    seed: int = RANDOM_SEED,
    policy: Optional[Callable] = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customers = generate_customers(n, rng)
    context = assign_context(customers, rng)
    promos = assign_promotions(n, rng, policy=policy)

    df = pd.concat([customers, context, promos], axis=1)
    df.insert(0, "round_number", np.arange(1, n + 1))
    df.insert(0, "transaction_id", [f"TXN_{i:05d}" for i in range(1, n + 1)])

    df = simulate_conversion(df, rng)
    df = simulate_basket_and_profit(df, rng)
    df["promotion_response_time"] = _simulate_response_time(df, rng)

    col_order = [
        "transaction_id",
        "round_number",
        "customer_id",
        "promotion_shown",
        "discount_rate",
        "conversion",
        "basket_size",
        "base_price",
        "price_charged",
        "cost",
        "contribution_profit",
        "order_method",
        "dine_in_takeout",
        "time_of_day",
        "day_of_week",
        "is_finals_week",
        "customer_type",
        "loyalty_membership",
        "weather",
        "queue_length",
        "repeat_visit_this_week",
        "promotion_response_time",
    ]
    return df[col_order]


def save_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    data_dict: pd.DataFrame,
    output_dir: str = ".",
) -> None:
    df.to_csv(os.path.join(output_dir, "campus_coffee_mab_synthetic.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "promotion_arm_summary.csv"), index=False)
    data_dict.to_csv(os.path.join(output_dir, "data_dictionary.csv"), index=False)
    _make_plots(df, output_dir)


def main() -> None:
    print("Generating campus coffee MAB synthetic dataset...")
    df = generate_dataset()
    checks = validate_dataset(df)
    summary = summarize_dataset(df)
    data_dict = build_data_dictionary()

    print("\n=== Validation checks ===")
    for k, v in checks.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")

    _print_ground_truth(df, summary)

    print("\n=== Overall dataset stats ===")
    print(f"  Transactions: {len(df):,}")
    print(f"  Overall conversion rate: {df['conversion'].mean():.3f}")
    print(f"  Mean contribution profit (all rows): ${df['contribution_profit'].mean():.3f}")
    print(f"  Unique customers: {df['customer_id'].nunique():,}")

    save_outputs(df, summary, data_dict)
    print("\nSaved: campus_coffee_mab_synthetic.csv, promotion_arm_summary.csv,")
    print("       data_dictionary.csv, conversion_by_arm.png, profit_by_arm.png,")
    print("       conversion_by_context.png")


if __name__ == "__main__":
    main()
