# Campus Coffee MAB — Synthetic Pricing Data

Synthetic transaction data for a **University of Washington campus coffee shop** testing off-peak promotions via app, kiosk, or counter. Each row is one visit (purchase or not). Promotions are **multi-armed bandit arms**; the reward for learning is **`contribution_profit`** (revenue minus COGS, 0 if no purchase).

## Promotion arms

| Arm | Role |
|-----|------|
| `no_discount` | Baseline margin, lower conversion |
| `discount_5` / `discount_10` / `discount_15` | Trade margin for conversion (10% often best on profit) |
| `bundle` | Higher basket size; sensitive to long queues |

**Round 1 setup:** promotions are assigned **uniformly at random** (static experiment) so true arm performance can be estimated. The generator supports plugging in adaptive policies later (ε-greedy, UCB, etc.).

## What’s in the repo

| File | Description |
|------|-------------|
| `generate_campus_coffee_mab_dataset.py` | Data generator (logistic conversion + basket/pricing) |
| `campus_coffee_mab_synthetic.csv` | 10,000 rows, main analysis file |
| `promotion_arm_summary.csv` | Mean conversion & profit by arm |
| `data_dictionary.csv` | Column definitions |
| `conversion_by_arm.png`, `profit_by_arm.png`, `conversion_by_context.png` | Quick validation plots |

## Quick start

```bash
pip install -r requirements.txt
python generate_campus_coffee_mab_dataset.py
```

Reproducible seed: `42`. Regenerating overwrites the CSVs and plots.

## Behavior baked into the data

- **Segments:** new (price-sensitive), returning, loyalty (higher conversion, more app use).
- **Context:** time of day (afternoon = off-peak), finals week, weather, queue length, order channel.
- **Interactions:** discounts help more in afternoon; bundle helps loyalty members but hurts when the queue is long.

## Ground truth (random assignment, current run)

| Arm | Mean conversion | Mean contribution profit |
|-----|-----------------|--------------------------|
| discount_10 | 0.50 | **$1.90** (best) |
| no_discount | 0.42 | $1.89 |
| discount_5 | 0.45 | $1.71 |
| bundle | 0.44 | $1.57 |
| discount_15 | 0.51 | $1.55 |

Use this file for static arm comparison; simulate **regret** by re-running assignment with bandit policies in `assign_promotions(..., policy=...)`.

## Downstream use

Compare policies (random, explore-then-commit, ε-greedy, UCB) on cumulative `contribution_profit` and regret vs. the best arm (`discount_10` in this draw). Column specs: see `data_dictionary.csv`.
