import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def generate_shap_plots():
    art_dir = Path("/Users/rafsan_temp/Library/CloudStorage/OneDrive-SeattleUniversity/SU Projects/PD Prediction Old/Mobility_Decline_Risk_Prediction_For_PD_Patients-explanation_llm_v2/revision_work/explanation_artifacts")
    repo_handoff_dir = Path("/Users/rafsan_temp/Library/CloudStorage/OneDrive-SeattleUniversity/SU Projects/PD Prediction Old/Mobility_Decline_Risk_Prediction_For_PD_Patients-explanation_llm_v2/github_repo_export/llm_layer_handoff")
    
    # Load feature map and create renaming dictionary
    feature_map_path = repo_handoff_dir / "feature_map.csv"
    feature_map_df = pd.read_csv(feature_map_path)
    rename_dict = dict(zip(feature_map_df['feature_name'], feature_map_df['short_name']))

    # Override specific names just for the plot visuals
    for k, v in rename_dict.items():
        if v == "FOG (Freezing of Gait)":
            rename_dict[k] = "Freezing of Gait"

    # Load data
    X_test = pd.read_csv(art_dir / "X_test.csv")
    
    # Rename columns to human-readable names
    X_test.rename(columns=rename_dict, inplace=True)
    
    s1_shap = np.load(art_dir / "shap_values_stage1.npy")
    s2_shap = np.load(art_dir / "shap_values_stage2.npy")
    routing_mask = np.load(art_dir / "routing_mask.npy")
    
    # We need X_test for stage 2 specifically for the plot
    # Stage 2 shap is only for routed patients.
    X_test_s2 = X_test[routing_mask]
    
    # Plot Stage 1
    plt.figure(figsize=(10, 8))
    shap.summary_plot(s1_shap, X_test, max_display=X_test.shape[1], show=False)
    plt.tight_layout()
    plt.savefig(art_dir / "shap_summary_stage1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Stage 2
    plt.figure(figsize=(10, 8))
    shap.summary_plot(s2_shap, X_test_s2, max_display=X_test_s2.shape[1], show=False)
    plt.tight_layout()
    plt.savefig(art_dir / "shap_summary_stage2.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("SHAP plots saved successfully to artifacts directory.")

if __name__ == "__main__":
    generate_shap_plots()
