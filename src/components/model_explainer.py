import shap
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


def explain_model():

    # Load model
    model = pickle.load(open("artifacts/model.pkl", "rb"))

    # Load data
    data = pd.read_csv("artifacts/train.csv")

    # Drop target
    if "Target" in data.columns:
        y = data["Target"]
        X = data.drop("Target", axis=1)
    else:
        X = data

    # Remove non-numeric columns automatically
    X = X.select_dtypes(include=["number"])

    os.makedirs("artifacts", exist_ok=True)

    model_name = type(model).__name__
    print("Loaded model:", model_name)

    # Sample small dataset for SHAP
    X_sample = X.sample(min(100, len(X)), random_state=42)

    # SHAP explainer
    explainer = shap.KernelExplainer(model.predict, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # Plot
    shap.summary_plot(shap_values, X_sample, show=False)

    plt.savefig("artifacts/shap_summary.png")

    print("SHAP plot saved to artifacts/shap_summary.png")


if __name__ == "__main__":
    explain_model()