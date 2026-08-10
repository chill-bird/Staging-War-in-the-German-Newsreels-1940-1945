import pickle
import numpy as np
import os
import glob

feature_dir = "/work/xb27qenu-ca_lmske/clip_features"
feature_files = glob.glob(os.path.join(feature_dir, "*.pkl"))

for fpath in feature_files:
    print(f"Processing {fpath} ...")
    
    with open(fpath, "rb") as f:
        features = pickle.load(f)

    # If features is a dict, extract the actual vectors
    if isinstance(features, dict):
        if "features" in features:
            features_array = np.asarray(features["features"], dtype=np.float32)
        else:
            raise ValueError(f"No 'features' key found in {fpath}")
    elif isinstance(features, np.ndarray) and features.shape == ():
        features_array = np.asarray(features.item(), dtype=np.float32)
    else:
        features_array = np.asarray(features, dtype=np.float32)

    print(f"New shape: {features_array.shape}")
    
    # Save to new file
    base, ext = os.path.splitext(fpath)
    new_fpath = base + "_fixed" + ext

    with open(new_fpath, "wb") as f:
        pickle.dump(features_array, f)

    print(f"Saved fixed features to {new_fpath}\n")
