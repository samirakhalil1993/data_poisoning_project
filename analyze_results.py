"""
Analysera och visualisera resultat från data poisoning experiment
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Filepaths
BASELINE_CSV = "str/results/logs/baseline.csv"
FLIP_CSV = "str/results/logs/flip.csv"
BACKDOOR_CSV = "str/results/logs/back_door.csv"
DEFENSE_CSV = "str/results/logs/defense_flip.csv"

def load_results():
    """Ladda alla resultat-filer"""
    results = {}
    
    if os.path.exists(BASELINE_CSV):
        results['baseline'] = pd.read_csv(BASELINE_CSV)
        print("✔ Baseline loaded")
    
    if os.path.exists(FLIP_CSV):
        results['flip'] = pd.read_csv(FLIP_CSV)
        print("✔ Label-flipping loaded")
    
    if os.path.exists(BACKDOOR_CSV):
        results['backdoor'] = pd.read_csv(BACKDOOR_CSV)
        print("✔ Backdoor loaded")
    
    if os.path.exists(DEFENSE_CSV):
        results['defense'] = pd.read_csv(DEFENSE_CSV)
        print("✔ Defense loaded")
    
    return results

def print_summary(results):
    """Skriv ut sammanfattning av resultat"""
    print("\n" + "="*60)
    print("RESULTAT SAMMANFATTNING")
    print("="*60)
    
    if 'baseline' in results:
        baseline_acc = results['baseline']['accuracy'].iloc[0]
        print(f"\n📊 Baseline Accuracy: {baseline_acc:.4f}")
    
    if 'flip' in results:
        print("\n📊 Label-Flipping Attack:")
        flip_df = results['flip'][['attack_rate', 'accuracy', 'f1']].sort_values('attack_rate')
        for _, row in flip_df.iterrows():
            print(f"  {row['attack_rate']*100:5.1f}% poison → Accuracy: {row['accuracy']:.4f}, F1: {row['f1']:.4f}")
    
    if 'backdoor' in results:
        print("\n📊 Backdoor Attack:")
        # Filtrera clean och trigger resultat
        clean_df = results['backdoor'][results['backdoor']['attack_type'] == 'backdoor_clean']
        trigger_df = results['backdoor'][results['backdoor']['attack_type'] == 'backdoor_trigger']
        
        for rate in sorted(clean_df['attack_rate'].unique()):
            clean_acc = clean_df[clean_df['attack_rate'] == rate]['accuracy'].iloc[0]
            trigger_row = trigger_df[trigger_df['attack_rate'] == rate]
            
            if not trigger_row.empty:
                trigger_acc = trigger_row['accuracy'].iloc[0]
                asr = trigger_row['ASR'].iloc[0] if 'ASR' in trigger_row.columns else None
                
                print(f"  {rate*100:5.1f}% poison → Clean: {clean_acc:.4f}, Trigger: {trigger_acc:.4f}, ASR: {asr:.4f}" if asr else f"  {rate*100:5.1f}% poison → Clean: {clean_acc:.4f}, Trigger: {trigger_acc:.4f}")
    
    if 'defense' in results:
        print("\n📊 Defense (IsolationForest):")
        defense_df = results['defense'][['attack_rate', 'accuracy', 'removed_count']].sort_values('attack_rate')
        for _, row in defense_df.iterrows():
            removed = row['removed_count'] if pd.notna(row['removed_count']) else 0
            print(f"  {row['attack_rate']*100:5.1f}% poison → Accuracy: {row['accuracy']:.4f} (removed {removed} examples)")

def plot_results(results):
    """Skapa visualiseringar"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Poisoning Experiment Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Label-Flipping - Accuracy vs Poison Rate
    if 'flip' in results and 'baseline' in results:
        ax = axes[0, 0]
        flip_df = results['flip'].sort_values('attack_rate')
        baseline_acc = results['baseline']['accuracy'].iloc[0]
        
        ax.plot(flip_df['attack_rate']*100, flip_df['accuracy'], 'o-', linewidth=2, markersize=8, label='Label-Flipping')
        ax.axhline(y=baseline_acc, color='g', linestyle='--', linewidth=2, label='Baseline')
        ax.set_xlabel('Poison Rate (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Label-Flipping Attack', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])
    
    # Plot 2: Backdoor - Clean vs Trigger Accuracy
    if 'backdoor' in results:
        ax = axes[0, 1]
        clean_df = results['backdoor'][results['backdoor']['attack_type'] == 'backdoor_clean'].sort_values('attack_rate')
        trigger_df = results['backdoor'][results['backdoor']['attack_type'] == 'backdoor_trigger'].sort_values('attack_rate')
        
        ax.plot(clean_df['attack_rate']*100, clean_df['accuracy'], 'o-', linewidth=2, markersize=8, label='Clean Test')
        ax.plot(trigger_df['attack_rate']*100, trigger_df['accuracy'], 's-', linewidth=2, markersize=8, label='Trigger Test')
        ax.set_xlabel('Poison Rate (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Backdoor Attack', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])
    
    # Plot 3: Attack Success Rate (ASR) for Backdoor
    if 'backdoor' in results:
        ax = axes[1, 0]
        trigger_df = results['backdoor'][results['backdoor']['attack_type'] == 'backdoor_trigger'].sort_values('attack_rate')
        
        if 'ASR' in trigger_df.columns:
            ax.plot(trigger_df['attack_rate']*100, trigger_df['ASR'], 'd-', linewidth=2, markersize=8, color='red', label='ASR')
            ax.set_xlabel('Poison Rate (%)', fontsize=12)
            ax.set_ylabel('Attack Success Rate', fontsize=12)
            ax.set_title('Backdoor - Attack Success Rate', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([0, 1.05])
    
    # Plot 4: Defense Effectiveness
    if 'defense' in results and 'flip' in results and 'baseline' in results:
        ax = axes[1, 1]
        
        # Jämför attacked vs defended
        flip_df = results['flip'].sort_values('attack_rate')
        defense_df = results['defense'].sort_values('attack_rate')
        baseline_acc = results['baseline']['accuracy'].iloc[0]
        
        # Hitta matchande rates
        for rate in defense_df['attack_rate'].unique():
            flip_acc = flip_df[flip_df['attack_rate'] == rate]['accuracy'].values
            defense_acc = defense_df[defense_df['attack_rate'] == rate]['accuracy'].values
            
            if len(flip_acc) > 0 and len(defense_acc) > 0:
                ax.plot([rate*100], [flip_acc[0]], 'ro', markersize=10, label='Attacked' if rate == defense_df['attack_rate'].iloc[0] else '')
                ax.plot([rate*100], [defense_acc[0]], 'go', markersize=10, label='Defended' if rate == defense_df['attack_rate'].iloc[0] else '')
                ax.plot([rate*100, rate*100], [flip_acc[0], defense_acc[0]], 'b-', alpha=0.5)
        
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2, label='Baseline')
        ax.set_xlabel('Poison Rate (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Defense Effectiveness', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Spara figur
    output_path = "str/results/experiment_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✔ Figur sparad: {output_path}")
    
    plt.show()

def main():
    print("Data Poisoning - Resultatanalys")
    print("="*60)
    
    # Ladda resultat
    results = load_results()
    
    if not results:
        print("\n❌ Inga resultat hittades. Kör experiment först!")
        return
    
    # Skriv ut sammanfattning
    print_summary(results)
    
    # Skapa visualiseringar
    print("\n📊 Skapar visualiseringar...")
    plot_results(results)
    
    print("\n✔ Analys klar!")

if __name__ == "__main__":
    main()
