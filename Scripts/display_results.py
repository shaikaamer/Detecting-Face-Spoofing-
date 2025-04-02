import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load reconstruction error results from CSV
csv_path = "reconstruction_errors.csv"
df = pd.read_csv(csv_path)

# Ensure labels are stored as integers (Live = 0, Spoof = 1)
df["true_label"] = df["true_label"].astype(int)

# mapping numeric labels to class names
df["label_name"] = df["true_label"].map({0: "Live", 1: "Spoof"})

# Visualization: Boxplot of reconstruction errors by class 
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="label_name", y="reconstruction_error", palette="Set2")

plt.title("🔍 Reconstruction Error Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Reconstruction Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("reconstruction_error_distribution.png")
plt.show()

# Descriptive Stats: Print reconstruction error stats per class 
print("\n📊 Reconstruction Error Summary:")
for label, group in df.groupby("label_name"):
    print(f"  {label}:")
    print(f"    Mean Error  = {group['reconstruction_error'].mean():.5f}")
    print(f"    Std Dev     = {group['reconstruction_error'].std():.5f}")
    print(f"    Max Error   = {group['reconstruction_error'].max():.5f}")

# Tabular Summary: Save and display full breakdown per class 
print("\n📁 Tabular Summary of Reconstruction Errors:")
summary = df.groupby("label_name")["reconstruction_error"].agg(["mean", "std", "max", "min", "count"]).round(5)
summary.to_csv("reconstruction_error_summary.csv")

print(summary)
