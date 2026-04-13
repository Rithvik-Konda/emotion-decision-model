"""
Simple simulation of emotional appraisal in decision making under uncertainty.

This script compares two conditions:
1. Constraint satisfaction with emotional appraisal
2. Constraint satisfaction without emotional appraisal

"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Scenario:
    name: str
    options: List[str]
    initial_activation: Dict[str, float]
    compatibility: Dict[Tuple[str, str], float]
    emotional_valence: Dict[str, float]


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def run_model(
    scenario: Scenario,
    alpha: float = 0.35,
    decay: float = 0.10,
    max_steps: int = 50,
    tolerance: float = 1e-4,
    use_emotion: bool = True,
):
    activations = dict(scenario.initial_activation)
    history = [dict(activations)]

    for step in range(1, max_steps + 1):
        new_activations = {}
        max_change = 0.0

        for option in scenario.options:
            support = 0.0
            for other in scenario.options:
                if other == option:
                    continue
                weight = scenario.compatibility.get((option, other), 0.0)
                support += weight * activations[other]

            emotion_term = alpha * scenario.emotional_valence.get(option, 0.0) if use_emotion else 0.0
            updated = activations[option] + support + emotion_term - decay * activations[option]
            updated = clamp(updated)
            new_activations[option] = updated
            max_change = max(max_change, abs(updated - activations[option]))

        activations = new_activations
        history.append(dict(activations))

        if max_change < tolerance:
            return {
                "final_activations": activations,
                "steps": step,
                "winner": max(activations, key=activations.get),
                "history": history,
                "converged": True,
            }

    return {
        "final_activations": activations,
        "steps": max_steps,
        "winner": max(activations, key=activations.get),
        "history": history,
        "converged": False,
    }


def make_scenarios():
    return [
        Scenario(
            name="Career choice under uncertainty",
            options=["Stable Job", "Startup", "Graduate School"],
            initial_activation={
                "Stable Job": 0.10,
                "Startup": 0.10,
                "Graduate School": 0.10,
            },
            compatibility={
                ("Stable Job", "Startup"): -0.30,
                ("Startup", "Stable Job"): -0.30,
                ("Stable Job", "Graduate School"): -0.10,
                ("Graduate School", "Stable Job"): -0.10,
                ("Startup", "Graduate School"): 0.20,
                ("Graduate School", "Startup"): 0.20,
            },
            emotional_valence={
                "Stable Job": 0.20,
                "Startup": 0.75,
                "Graduate School": 0.10,
            },
        ),
        Scenario(
            name="Medical decision with risk tradeoff",
            options=["Immediate Surgery", "Medication", "Watchful Waiting"],
            initial_activation={
                "Immediate Surgery": 0.10,
                "Medication": 0.10,
                "Watchful Waiting": 0.10,
            },
            compatibility={
                ("Immediate Surgery", "Medication"): -0.25,
                ("Medication", "Immediate Surgery"): -0.25,
                ("Immediate Surgery", "Watchful Waiting"): -0.35,
                ("Watchful Waiting", "Immediate Surgery"): -0.35,
                ("Medication", "Watchful Waiting"): 0.10,
                ("Watchful Waiting", "Medication"): 0.10,
            },
            emotional_valence={
                "Immediate Surgery": -0.40,
                "Medication": 0.35,
                "Watchful Waiting": -0.15,
            },
        ),
        Scenario(
            name="Travel planning with conflicting preferences",
            options=["Cheap Flight", "Direct Flight", "Train"],
            initial_activation={
                "Cheap Flight": 0.10,
                "Direct Flight": 0.10,
                "Train": 0.10,
            },
            compatibility={
                ("Cheap Flight", "Direct Flight"): -0.20,
                ("Direct Flight", "Cheap Flight"): -0.20,
                ("Cheap Flight", "Train"): 0.05,
                ("Train", "Cheap Flight"): 0.05,
                ("Direct Flight", "Train"): -0.10,
                ("Train", "Direct Flight"): -0.10,
            },
            emotional_valence={
                "Cheap Flight": 0.10,
                "Direct Flight": 0.45,
                "Train": 0.25,
            },
        ),
    ]


def main():
    scenarios = make_scenarios()
    print("Emotion and Decision Making Simulation")
    print("=" * 40)

    for scenario in scenarios:
        with_emotion = run_model(scenario, use_emotion=True)
        without_emotion = run_model(scenario, use_emotion=False)

        print(f"\nScenario: {scenario.name}")
        print("-" * 40)
        print("With emotion")
        print(f"Winner: {with_emotion['winner']}")
        print(f"Steps to convergence: {with_emotion['steps']}")
        print(f"Final activations: {with_emotion['final_activations']}")
        print("\nWithout emotion")
        print(f"Winner: {without_emotion['winner']}")
        print(f"Steps to convergence: {without_emotion['steps']}")
        print(f"Final activations: {without_emotion['final_activations']}")


if __name__ == "__main__":
    main()
