import pandas as pd

# Load the CSV containing reconstruction errors and true labels
df = pd.read_csv("reconstruction_errors.csv")

# Map numeric labels to readable class names (Live = 0, Spoof = 1)
df["label_name"] = df["true_label"].map({0: "Live", 1: "Spoof"})

# Compute reconstruction error statistics grouped by class
# - This provides insight into how well the autoencoder distinguishes live vs spoof
print("\nTabular Summary of Reconstruction Errors:")
summary = df.groupby("label_name")["reconstruction_error"].agg(["mean", "std", "max", "min", "count"]).round(5)
print(summary)

#  Save the summary as a CSV 
summary.to_csv("reconstruction_error_summary.csv")
print("Saved as reconstruction_error_summary.csv")
