"""
Simple multi-armed bandit policies for continuous rewards.

Policies are non-contextual (same decision rule for all contexts).
Arms are strings (e.g., "discount_10").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


class BanditPolicy:
    name: str

    def select_arm(self, t: int, rng: np.random.Generator) -> str:  # pragma: no cover
        raise NotImplementedError

    def update(self, arm: str, reward: float) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class _RunningMean:
    n: int = 0
    mean: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n


class RandomPolicy(BanditPolicy):
    def __init__(self, arms: List[str]):
        self.name = "random"
        self.arms = list(arms)

    def select_arm(self, t: int, rng: np.random.Generator) -> str:
        return str(rng.choice(self.arms))

    def update(self, arm: str, reward: float) -> None:
        return


class ExploreThenCommitPolicy(BanditPolicy):
    """Explore each arm m_per_arm times, then commit to best mean reward arm."""

    def __init__(self, arms: List[str], m_per_arm: int = 200):
        self.name = f"etc_m{int(m_per_arm)}"
        self.arms = list(arms)
        self.m_per_arm = int(m_per_arm)
        self.stats: Dict[str, _RunningMean] = {a: _RunningMean() for a in self.arms}
        self._committed_arm: Optional[str] = None

    def select_arm(self, t: int, rng: np.random.Generator) -> str:
        if self._committed_arm is not None:
            return self._committed_arm

        for a in self.arms:
            if self.stats[a].n < self.m_per_arm:
                return a

        self._committed_arm = max(self.arms, key=lambda a: self.stats[a].mean)
        return self._committed_arm

    def update(self, arm: str, reward: float) -> None:
        self.stats[arm].update(float(reward))


class EpsilonGreedyPolicy(BanditPolicy):
    def __init__(self, arms: List[str], epsilon: float = 0.10, warm_start: int = 1):
        self.name = f"eps_greedy_{epsilon:.2f}"
        self.arms = list(arms)
        self.epsilon = float(epsilon)
        self.warm_start = int(warm_start)
        self.stats: Dict[str, _RunningMean] = {a: _RunningMean() for a in self.arms}

    def select_arm(self, t: int, rng: np.random.Generator) -> str:
        for a in self.arms:
            if self.stats[a].n < self.warm_start:
                return a

        if rng.random() < self.epsilon:
            return str(rng.choice(self.arms))

        return max(self.arms, key=lambda a: self.stats[a].mean)

    def update(self, arm: str, reward: float) -> None:
        self.stats[arm].update(float(reward))


class UCB1Policy(BanditPolicy):
    """UCB1 index: mean + c * sqrt(2 ln t / n)."""

    def __init__(self, arms: List[str], c: float = 1.0):
        self.name = f"ucb1_c{c:.2f}"
        self.arms = list(arms)
        self.c = float(c)
        self.stats: Dict[str, _RunningMean] = {a: _RunningMean() for a in self.arms}

    def select_arm(self, t: int, rng: np.random.Generator) -> str:
        for a in self.arms:
            if self.stats[a].n == 0:
                return a

        ln_t = np.log(max(2, t))

        def index(a: str) -> float:
            s = self.stats[a]
            bonus = self.c * np.sqrt(2.0 * ln_t / s.n)
            return s.mean + bonus

        return max(self.arms, key=index)

    def update(self, arm: str, reward: float) -> None:
        self.stats[arm].update(float(reward))

