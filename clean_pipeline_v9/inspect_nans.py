import numpy as np

path = "pod_test/simulations/sim_data.npz"
print(f"Loading {path}...")
data = np.load(path, allow_pickle=True)
X = data["X"]
Theta = data["Theta"]
theta_keys = data["theta_keys"]

print(f"Shapes - X: {X.shape}, Theta: {Theta.shape}")

# Check Theta (Parameters)
if np.isnan(Theta).any() or np.isinf(Theta).any():
    print("\n[!] Found NaNs/Infs in THETA (Parameters)!")
else:
    print("\n[OK] Theta (Parameters) looks clean.")

# Check X (Summaries)
bad_indices = np.where(~np.isfinite(X))
if len(bad_indices[0]) > 0:
    print("\n[!] Found NaNs/Infs in X (Summary Statistics)!")
    bad_cols = np.unique(bad_indices[1])
    print(f"  - Affected column indices: {bad_cols}")
    print("  - Sample bad values (Row 0):")
    for c in bad_cols[:5]:
        print(f"    Col {c}: {X[0, c]}")
else:
    print("\n[OK] X (Summaries) looks clean.")