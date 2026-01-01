import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
CSV_PATH = r"C:\Users\lenovo\Desktop\image classicification\idrid_labels.csv"
IMAGE_DIR = r"C:\Users\lenovo\Desktop\image classicification\Imagenes"
OUTPUT_DIR = r"C:\Users\lenovo\Desktop\image classicification\dataset"
SEED = 42
# ----------------------------------------

df = pd.read_csv(CSV_PATH)

# Drop junk columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Clean id_code
df["id_code"] = (
    df["id_code"]
    .astype(str)
    .str.strip()
    .str.replace("test", "", regex=False)
)

# Create filename + label
df["filename"] = df["id_code"] + ".jpg"
df["label"] = df["diagnosis"]

# Verify existence
df["exists"] = df["filename"].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, x))
)

# Keep only valid images
df = df[df["exists"]].reset_index(drop=True)

# ---------------- SPLIT ----------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED
)

# ---------------- COPY ----------------
def copy_images(dataframe, split):
    for _, row in dataframe.iterrows():
        src = os.path.join(IMAGE_DIR, row["filename"])
        dst = os.path.join(OUTPUT_DIR, split, str(row["label"]))
        os.makedirs(dst, exist_ok=True)
        shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("✅ DR dataset prepared successfully")
