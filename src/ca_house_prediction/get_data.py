import os

import kaggle

# Ensure Kaggle API is configured
# kaggle.json must be in ~/.kaggle/ or C:\Users\<YourUsername>\.kaggle\

try:
    dataset = "camnugent/california-housing-prices"
    output_dir = os.path.join(
        os.path.abspath("../../"), "artifacts", "ca_house_prediction", "data"
    )
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download dataset
    print(f"Downloading {dataset} to {output_dir}...")
    kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print("Download complete. Files are in:", output_dir)
except Exception as e:
    print(f"Error: {e}")
