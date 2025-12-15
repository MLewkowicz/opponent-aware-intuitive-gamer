import itertools
from typing import Dict
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any
import math
class Experiment():
    def __init__(self, policies: Dict[str, Any], sampler: Any):
        self.policies = policies
        self.sampler = sampler

    def run_pairwise_comparison(self):
        comparison_results = {}
        
        # Get samples with error handling
        try:
            samples = self.sampler.samples()
            if samples is None:
                raise ValueError("Sampler returned None")
            print(f"Got {len(samples)} samples for analysis")
        except Exception as e:
            print(f"Error getting samples: {e}")
            return {}

        for game_state in samples:
            state = game_state['state'] 
            legal_actions = list(state.legal_actions())

            policy_probs = {
                name: p['policy'].action_likelihoods(state) 
                for name, p in self.policies.items()
            }

            for (name_a, probs_a), (name_b, probs_b) in itertools.combinations_with_replacement(policy_probs.items(), 2):
                
                pair_key = (name_a, name_b)
                if 'random' in pair_key:
                    continue

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

        samples = self.sampler.samples()
        for game_state in samples:
            state = game_state['state'] 
            
            policy_max_actions = {}
            for name, policy in self.policies.items():
                probs = policy['policy'].action_likelihoods(state)
                    
                if probs:
                    best_action = max(probs, key=probs.get)
                    policy_max_actions[name] = (best_action, state)
                else:
                    policy_max_actions[name] = None

            for (name_a, sa_a), (name_b, sa_b) in itertools.combinations_with_replacement(policy_max_actions.items(), 2):
                action_a, state = sa_a if sa_a else (None, None)
                action_b, _ = sa_b if sa_b else (None, None)
                pair_key = (name_a, name_b)
                if pair_key not in comparison_results:
                    comparison_results[pair_key] = {"agreements": 0, "totals": 0}

                if action_a == action_b:
                    comparison_results[pair_key]["agreements"] += 1
                # else:
                #     print("----")
                #     print(state)
                #     print(state.current_player())
                #     print(name_a, action_a)
                #     print(name_b, action_b)
                #     print("----")
                comparison_results[pair_key]["totals"] += 1
        return comparison_results

def visualize_agreement_heatmap(results_data, titles=None):
    """
    Visualizes agreement results. Accepts a single dict or a list of dicts.
    """
    # 1. Normalize inputs to lists
    if isinstance(results_data, dict):
        results_list = [results_data]
    else:
        results_list = results_data

    # Handle titles
    if titles is None:
        titles = [f"Comparison {i+1}" for i in range(len(results_list))]
    elif isinstance(titles, str):
        titles = [titles]

    n_plots = len(results_list)
    
    # 2. Setup Figure and Axes
    if n_plots == 1:
        # Simple single plot
        fig, axes = plt.subplots(figsize=(8, 6))
        axes_list = [axes] # Wrap in list to make iterable
    else:
        cols = min(n_plots, 3)
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
        axes_list = axes.flatten()

    # 3. Iterate and Plot
    for i, (results, ax) in enumerate(zip(results_list, axes_list)):
        # Extract unique policy names
        policy_names = sorted(list({p for pair in results.keys() for p in pair}))
        n = len(policy_names)
        
        if n == 0:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        # Build DataFrame
        df = pd.DataFrame(np.zeros((n, n)), index=policy_names, columns=policy_names)
        
        for (name_a, name_b), stats in results.items():
            # Support both 'totals' and 'total' keys
            total = stats.get("totals", stats.get("total", 0))
            score = stats["agreements"] / total if total > 0 else 0
            
            # Fill symmetric matrix
            if name_a in df.index and name_b in df.columns:
                df.at[name_a, name_b] = score
                df.at[name_b, name_a] = score
                # Fill diagonal with 1.0 (100%) for visual completeness
                if name_a == name_b: 
                     df.at[name_a, name_b] = 1.0

        # Plot Heatmap
        # Mask the upper triangle to reduce visual clutter (optional, remove mask=mask if unwanted)
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
        
        sns.heatmap(df, ax=ax, annot=True, fmt=".1%", cmap="Blues", 
                    vmin=0, vmax=1.0, square=True, mask=None,
                    cbar_kws={'shrink': 0.8})
        
        current_title = titles[i] if i < len(titles) else f"Plot {i+1}"
        ax.set_title(current_title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    for j in range(n_plots, len(axes_list)):
        axes_list[j].axis('off')

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