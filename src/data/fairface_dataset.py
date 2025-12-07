import os
import pandas as pd
from src.preprocessing.face_processing import compute_skin_ita_from_image


def load_image_table(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV that has at least columns: image_path, label
    - image_path: absolute or relative paths to images
    - label: e.g., 'warm', 'cool', 'neutral' or seasons
    """
    df = pd.read_csv(csv_path)
    required_cols = {"image_path", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each image in the table, compute ITA and return a new DataFrame
    with columns: ita, label
    (Later we can add more features like hair and eye color.)
    """
    ita_values = []
    missing_paths = []

    for path in df["image_path"]:
        if not os.path.exists(path):
            missing_paths.append(path)
            ita_values.append(float("nan"))
            continue

        ita = compute_skin_ita_from_image(path)
        ita_values.append(ita)

    feature_df = pd.DataFrame({
        "ita": ita_values,
        "label": df["label"].values
    })

    if missing_paths:
        print(f"Warning: {len(missing_paths)} image paths were missing.")

    return feature_df
