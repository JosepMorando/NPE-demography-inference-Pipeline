import numpy as np

def main():
    print("Loading simulations...")
    sim_data = np.load("pod_test/simulations/sim_data.npz", allow_pickle=True)
    X_sim = sim_data["X"]
    
    print("Loading observed data...")
    obs_data = np.load("observed_data/observed_summaries.npz", allow_pickle=True)
    # Handle different keys
    if "x_obs" in obs_data:
        x_obs = obs_data["x_obs"]
    elif "x" in obs_data:
        x_obs = obs_data["x"]
    else:
        x_obs = obs_data[obs_data.files[0]]
        
    x_obs = x_obs.flatten()

    print(f"\nDimensions: Sim {X_sim.shape}, Obs {x_obs.shape}")
    
    if X_sim.shape[1] != x_obs.shape[0]:
        print(f"ERROR: Dimension mismatch! Sim has {X_sim.shape[1]} stats, Obs has {x_obs.shape[0]}.")
        return

    print("\n--- OUTLIER CHECK ---")
    n_features = len(x_obs)
    n_outliers = 0
    
    # Define roughly what the features are based on your R script order
    # 0-K*bins: SFS
    # Next K: Het
    # ...
    
    print(f"{'Feature idx':<12} {'Sim Mean':<12} {'Sim Range':<25} {'Observed':<12} {'Status'}")
    print("-" * 75)
    
    for i in range(n_features):
        sim_vals = X_sim[:, i]
        valid_sims = sim_vals[np.isfinite(sim_vals)]
        
        if len(valid_sims) == 0:
            continue
            
        mn, mx = np.min(valid_sims), np.max(valid_sims)
        mean = np.mean(valid_sims)
        obs = x_obs[i]
        
        is_outlier = (obs < mn) or (obs > mx)
        
        # Show first 20 stats (SFS) and any outliers
        if is_outlier or i < 20 or (i % 50 == 0):
            status = "[OUTLIER !!!]" if is_outlier else "OK"
            if is_outlier: n_outliers += 1
            print(f"{i:<12} {mean:<12.5f} [{mn:.5f}, {mx:.5f}]     {obs:<12.5f} {status}")

    print("-" * 75)
    print(f"Total Outliers: {n_outliers} / {n_features} features are outside simulation range.")

if __name__ == "__main__":
    main()