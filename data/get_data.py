import os
import urllib.request
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal","target"]
RAW_PATH = "data/raw_cleveland.data"
OUT_PATH = "data/heart.csv"

def main():
    os.makedirs("data", exist_ok=True)
    try:
        print(f"Downloading from {URL} ...")
        urllib.request.urlretrieve(URL, RAW_PATH)
        print("Download successful.")
    except Exception as e:
        print(f"ERROR: download failed: {e}")
        return

    df = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, na_values="?")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with shape {df.shape}")

if __name__ == "__main__":
    main()