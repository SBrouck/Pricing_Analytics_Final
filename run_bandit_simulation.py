"""
Run online bandit simulations on the Campus Coffee synthetic environment.

This simulates policies interacting with the same behavioral model used in
`generate_campus_coffee_mab_dataset.py`, but in an online learning loop.

Outputs (bandit_outputs/ by default):
- bandit_summary.csv
- oracle_means.csv
- bandit_timeseries_mean.csv.gz (compressed)
- cumulative_profit_by_policy.png
- cumulative_regret_by_policy.png
- arm_share_over_time.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bandit_policies import (
    BanditPolicy,
    EpsilonGreedyPolicy,
    ExploreThenCommitPolicy,
    RandomPolicy,
    UCB1Policy,
)

# Reuse environment pieces from the generator
from generate_campus_coffee_mab_dataset import (  # noqa: E402
    DISCOUNT_RATE_MAP,
    PROMOTION_ARMS,
    assign_context,
    generate_customers,
    simulate_basket_and_profit,
    _compute_conversion_logit,
    _simulate_response_time,
)


@dataclass
class Oracle:
    mu_by_arm: Dict[str, float]

    @property
    def best_arm(self) -> str:
        return max(self.mu_by_arm, key=self.mu_by_arm.get)

    @property
    def mu_star(self) -> float:
        return float(self.mu_by_arm[self.best_arm])


def estimate_oracle_means(
    n_mc: int = 30_000, seed: int = 123, chunk: int = 15_000
) -> Oracle:
    """Monte Carlo estimate of expected reward per arm for pseudo-regret."""
    rng = np.random.default_rng(seed)
    sums = {a: 0.0 for a in PROMOTION_ARMS}
    counts = {a: 0 for a in PROMOTION_ARMS}

    remaining = n_mc
    while remaining > 0:
        n = min(chunk, remaining)
        remaining -= n
        done = n_mc - remaining
        print(f"[oracle] sampling {done:,}/{n_mc:,} contexts...")

        customers = generate_customers(n, rng)
        context = assign_context(customers, rng)
        base = pd.concat([customers, context], axis=1)

        for arm in PROMOTION_ARMS:
            df = base.copy()
            df["promotion_shown"] = arm
            df["discount_rate"] = DISCOUNT_RATE_MAP[arm]

            eta = _compute_conversion_logit(df)
            p = 1 / (1 + np.exp(-eta))
            df["conversion"] = rng.binomial(1, p)
            df = simulate_basket_and_profit(df, rng)

            rewards = df["contribution_profit"].to_numpy(dtype=float)
            sums[arm] += float(rewards.sum())
            counts[arm] += int(len(rewards))

    mu = {a: (sums[a] / counts[a]) for a in PROMOTION_ARMS}
    return Oracle(mu_by_arm=mu)


def _simulate_one_round(
    row_base: pd.Series, arm: str, rng: np.random.Generator
) -> Tuple[float, int]:
    """Return (reward, conversion) for a single round given base row + chosen arm."""
    df = row_base.to_frame().T.copy().reset_index(drop=True)
    df["promotion_shown"] = arm
    df["discount_rate"] = DISCOUNT_RATE_MAP[arm]

    eta_arr = _compute_conversion_logit(df)
    eta = float(np.asarray(eta_arr)[0])
    p = 1 / (1 + np.exp(-eta))
    conv = int(rng.binomial(1, p))
    df["conversion"] = conv

    df = simulate_basket_and_profit(df, rng)
    df["promotion_response_time"] = _simulate_response_time(df, rng)

    reward = float(df.loc[df.index[0], "contribution_profit"])
    return reward, conv


def run_single_simulation(policy: BanditPolicy, T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = generate_customers(T, rng)
    context = assign_context(customers, rng)
    base = pd.concat([customers, context], axis=1)

    chosen: List[str] = []
    rewards = np.zeros(T)
    conversions = np.zeros(T, dtype=int)

    for t in range(1, T + 1):
        arm = policy.select_arm(t=t, rng=rng)
        r, c = _simulate_one_round(base.iloc[t - 1], arm, rng)
        policy.update(arm, r)

        chosen.append(arm)
        rewards[t - 1] = r
        conversions[t - 1] = c

    return pd.DataFrame(
        {
            "t": np.arange(1, T + 1),
            "policy": policy.name,
            "arm": chosen,
            "reward": rewards,
            "conversion": conversions,
        }
    )


def run_experiments(
    T: int = 10_000,
    n_runs: int = 5,
    seed0: int = 2026,
    output_dir: str = "bandit_outputs",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    print("Estimating oracle arm means (Monte Carlo)...")
    oracle = estimate_oracle_means()
    print(
        f"[oracle] best_arm={oracle.best_arm} mu*={oracle.mu_star:.4f} | "
        + ", ".join([f"{a}={oracle.mu_by_arm[a]:.3f}" for a in PROMOTION_ARMS])
    )

    policies: List[BanditPolicy] = [
        RandomPolicy(PROMOTION_ARMS),
        ExploreThenCommitPolicy(PROMOTION_ARMS, m_per_arm=200),
        EpsilonGreedyPolicy(PROMOTION_ARMS, epsilon=0.10, warm_start=1),
        UCB1Policy(PROMOTION_ARMS, c=1.0),
    ]

    all_logs = []
    for i in range(n_runs):
        run_seed = seed0 + i
        print(f"[sim] run {i+1}/{n_runs} (seed={run_seed})")

        for pol in policies:
            # Fresh instance per run
            if pol.name.startswith("random"):
                policy = RandomPolicy(PROMOTION_ARMS)
            elif pol.name.startswith("etc_"):
                policy = ExploreThenCommitPolicy(PROMOTION_ARMS, m_per_arm=200)
            elif pol.name.startswith("eps_greedy"):
                policy = EpsilonGreedyPolicy(PROMOTION_ARMS, epsilon=0.10, warm_start=1)
            elif pol.name.startswith("ucb1"):
                policy = UCB1Policy(PROMOTION_ARMS, c=1.0)
            else:
                raise ValueError(f"Unknown policy template: {pol.name}")

            logs = run_single_simulation(
                policy=policy,
                T=T,
                seed=run_seed * 10 + (hash(pol.name) % 997),
            )
            logs["run"] = i
            all_logs.append(logs)

    df = pd.concat(all_logs, ignore_index=True)
    df["cum_reward"] = df.groupby(["run", "policy"])["reward"].cumsum()
    df["cum_regret"] = df.groupby(["run", "policy"])["reward"].cumsum().rsub(
        df["t"] * oracle.mu_star
    )

    # Summary table
    summary = (
        df.groupby(["policy", "run"])
        .agg(
            final_cum_profit=("cum_reward", "last"),
            final_cum_regret=("cum_regret", "last"),
            mean_reward=("reward", "mean"),
            conversion_rate=("conversion", "mean"),
        )
        .reset_index()
    )
    summary_mean = (
        summary.groupby("policy")
        .agg(
            avg_final_profit=("final_cum_profit", "mean"),
            sd_final_profit=("final_cum_profit", "std"),
            avg_final_regret=("final_cum_regret", "mean"),
            sd_final_regret=("final_cum_regret", "std"),
            avg_reward=("mean_reward", "mean"),
            avg_conversion=("conversion_rate", "mean"),
        )
        .reset_index()
        .sort_values("avg_final_profit", ascending=False)
    )
    summary_mean["oracle_best_arm"] = oracle.best_arm
    summary_mean["oracle_mu_star"] = oracle.mu_star
    summary_mean.to_csv(os.path.join(output_dir, "bandit_summary.csv"), index=False)

    # Oracle means
    pd.DataFrame(
        [{"promotion_shown": a, "oracle_mean_reward": oracle.mu_by_arm[a]} for a in PROMOTION_ARMS]
    ).to_csv(os.path.join(output_dir, "oracle_means.csv"), index=False)

    # Mean time series across runs (compressed)
    ts = (
        df.groupby(["policy", "t"])
        .agg(
            mean_cum_reward=("cum_reward", "mean"),
            mean_cum_regret=("cum_regret", "mean"),
        )
        .reset_index()
    )
    ts.to_csv(
        os.path.join(output_dir, "bandit_timeseries_mean.csv.gz"),
        index=False,
        compression="gzip",
    )

    # Plots (kept locally; gitignored)
    def _lineplot(ycol: str, title: str, ylabel: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=ts, x="t", y=ycol, hue="policy", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()

    _lineplot(
        "mean_cum_reward",
        "Cumulative contribution profit (mean across runs)",
        "Cumulative profit ($)",
        "cumulative_profit_by_policy.png",
    )
    _lineplot(
        "mean_cum_regret",
        "Cumulative pseudo-regret vs best fixed arm (mean across runs)",
        "Cumulative regret ($)",
        "cumulative_regret_by_policy.png",
    )

    df["window"] = ((df["t"] - 1) // 250) * 250 + 1
    share = (
        df.groupby(["policy", "run", "window", "arm"])
        .size()
        .reset_index(name="n")
    )
    share["share"] = share.groupby(["policy", "run", "window"])["n"].transform(
        lambda x: x / x.sum()
    )
    share_mean = share.groupby(["policy", "window", "arm"])["share"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=share_mean, x="window", y="share", hue="arm", style="policy", ax=ax)
    ax.set_title("Arm selection share over time (250-round windows)")
    ax.set_xlabel("Round (window start)")
    ax.set_ylabel("Selection share")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "arm_share_over_time.png"), dpi=150)
    plt.close()

    print(f"Saved bandit outputs to: {output_dir}/")
    print(f"Oracle best arm: {oracle.best_arm} (mu*={oracle.mu_star:.4f})")


if __name__ == "__main__":
    run_experiments()

