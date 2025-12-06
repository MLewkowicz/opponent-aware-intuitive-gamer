# Removed circular import - functions will be passed as parameters instead
import itertools
from typing import Dict
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any

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

        samples = self.sampler.samples()
        for game_state in samples:
            state = game_state['state'] 
            
            policy_max_actions = {}
            for name, policy in self.policies.items():
                probs = policy['policy'].action_likelihoods(state)
                
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
        if stats["totals"] > 0:
            score = stats["agreements"] / stats["totals"]
        else:
            score = 0.0
        df.at[name_a, name_b] = score
        df.at[name_b, name_a] = score

    # Create figure with better styling
    plt.figure(figsize=(10, 8))
    plt.style.use('default')  # Clean base style
    
    # Use a more pleasing color palette - blues with good contrast
    sns.heatmap(df, 
                annot=True, 
                fmt=".1%",  # Show one decimal place for percentages
                cmap="Blues",  # Easier on the eyes than RdYlGn
                vmin=0, 
                vmax=1.0,
                square=True,  # Make cells square for better symmetry
                linewidths=0.5,  # Add subtle grid lines
                linecolor='white',
                cbar_kws={
                    'label': 'Agreement Rate',
                    'shrink': 0.8,
                    'format': '%.0%'
                },
                annot_kws={'size': 12, 'weight': 'bold'})  # Larger, bolder text
    
    # Improve title and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Policy', fontsize=12, fontweight='bold')
    plt.ylabel('Policy', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
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