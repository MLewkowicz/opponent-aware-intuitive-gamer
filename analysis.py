from main import load_game, load_policies, load_sampler
import itertools
from typing import Dict
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Experiment():
    def __init__(self, num_samples: int, policies: Dict[str, Any], sampler: Any):
        self.num_samples = num_samples
        self.policies = policies
        self.sampler = sampler

    def run_pairwise_comparison(self):
        comparison_results = {}

        for _ in range(self.num_samples):
            state = self.sampler.sample()
            legal_actions = list(state.legal_actions())

            policy_probs = {
                name: p.action_likelihoods(state) 
                for name, p in self.policies.items()
            }

            for (name_a, probs_a), (name_b, probs_b) in itertools.combinations(policy_probs.items(), 2):
                
                pair_key = (name_a, name_b)
                if pair_key not in comparison_results:
                    comparison_results[pair_key] = {"agreements": 0, "totals": 0}

                for a1, a2 in itertools.combinations(legal_actions, 2):
                    diff_a = probs_a.get(a1, 0.0) - probs_a.get(a2, 0.0)
                    diff_b = probs_b.get(a1, 0.0) - probs_b.get(a2, 0.0)

                    if (diff_a * diff_b) > 0:
                        comparison_results[pair_key]["agreements"] += 1
                    comparison_results[pair_key]["totals"] += 1

        return comparison_results
    
    def run_max_action_disagreement(self):
        """
        Checks if the action with the highest probability (Max Action) 
        differs across policies.
        """
        comparison_results = {}

        for _ in range(self.num_samples):
            state = self.sampler.sample()
            
            policy_max_actions = {}
            for name, policy in self.policies.items():
                probs = policy.action_likelihoods(state)
                
                if probs:
                    best_action = max(probs, key=probs.get)
                    policy_max_actions[name] = best_action
                else:
                    policy_max_actions[name] = None

            for (name_a, action_a), (name_b, action_b) in itertools.combinations(policy_max_actions.items(), 2):
                
                pair_key = (name_a, name_b)
                if pair_key not in comparison_results:
                    comparison_results[pair_key] = {"agreements": 0, "totals": 0}

                if action_a == action_b:
                    comparison_results[pair_key]["agreements"] += 1
                comparison_results[pair_key]["totals"] += 1
        return comparison_results

def visualize_agreement_heatmap(comparison_results, title="Policy Agreement Scores"):
    """
    Takes the dictionary output from Experiment.run_* functions 
    and generates a heatmap of agreement percentages.
    """
    policy_names = set()
    for (p1, p2) in comparison_results.keys():
        policy_names.add(p1)
        policy_names.add(p2)
    
    labels = sorted(list(policy_names))
    n = len(labels)
    df = pd.DataFrame(np.ones((n, n)), index=labels, columns=labels)
    for (name_a, name_b), stats in comparison_results.items():
        if stats["total"] > 0:
            score = stats["agreements"] / stats["total"]
        else:
            score = 0.0
        df.at[name_a, name_b] = score
        df.at[name_b, name_a] = score

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2%", cmap="RdYlGn", vmin=0, vmax=1.0)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Mock data resembling your output
    mock_results = {
        ('MCTS', 'Random'): {'agreements': 60, 'total': 100},
        ('MCTS', 'Greedy'): {'agreements': 85, 'total': 100},
        ('Random', 'Greedy'): {'agreements': 40, 'total': 100},
    }

    visualize_agreement_heatmap(mock_results)