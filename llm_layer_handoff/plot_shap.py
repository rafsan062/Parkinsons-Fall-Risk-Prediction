import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_shap_plots():
    art_dir = Path("/Users/rafsan_temp/Library/CloudStorage/OneDrive-SeattleUniversity/SU Projects/PD Prediction Old/Mobility_Decline_Risk_Prediction_For_PD_Patients-explanation_llm_v2/revision_work/explanation_artifacts")
    
    # Load data
    X_test = pd.read_csv(art_dir / "X_test.csv")
    s1_shap = np.load(art_dir / "shap_values_stage1.npy")
    s2_shap = np.load(art_dir / "shap_values_stage2.npy")
    routing_mask = np.load(art_dir / "routing_mask.npy")
    
    # We need X_test for stage 2 specifically for the plot?
    # Stage 2 shap is only for routed patients.
    X_test_s2 = X_test[routing_mask]
    
    # Plot Stage 1
    plt.figure(figsize=(10, 8))
    shap.summary_plot(s1_shap, X_test, show=False)
    plt.tight_layout()
    plt.savefig(art_dir / "shap_summary_stage1.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Stage 2
    plt.figure(figsize=(10, 8))
    shap.summary_plot(s2_shap, X_test_s2, show=False)
    plt.tight_layout()
    plt.savefig(art_dir / "shap_summary_stage2.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("SHAP plots saved successfully.")

if __name__ == "__main__":
    generate_shap_plots()
