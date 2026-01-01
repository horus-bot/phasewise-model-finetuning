import pandas as pd

CSV_PATH = r"C:\Users\lenovo\Desktop\image classicification\idrid_labels.csv"

df = pd.read_csv(CSV_PATH)

print("\n=== BASIC INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n=== FIRST 10 ROWS ===")
print(df.head(10))

print("\n=== LAST 10 ROWS ===")
print(df.tail(10))

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== NULL VALUES PER COLUMN ===")
print(df.isnull().sum())

print("\n=== UNIQUE id_code COUNT ===")
print(df["id_code"].nunique())

print("\n=== SAMPLE id_code VALUES (RAW) ===")
print(df["id_code"].astype(str).head(20).tolist())

print("\n=== id_code VALUES CONTAINING 'test' ===")
test_ids = df[df["id_code"].astype(str).str.contains("test", case=False, na=False)]
print("Count:", len(test_ids))
print(test_ids["id_code"].astype(str).head(20).tolist())

print("\n=== LEADING / TRAILING SPACE CHECK ===")
space_ids = df[
    df["id_code"].astype(str).str.startswith(" ") |
    df["id_code"].astype(str).str.endswith(" ")
]
print("Count:", len(space_ids))
print(space_ids["id_code"].astype(str).head(10).tolist())

print("\n=== DIAGNOSIS DISTRIBUTION ===")
print(df["diagnosis"].value_counts().sort_index())

print("\n=== MACULAR EDEMA DISTRIBUTION ===")
if "Risk of macular edema" in df.columns:
    print(df["Risk of macular edema"].value_counts().sort_index())
else:
    print("Column not found")

print("\n=== DUPLICATE ROWS (by id_code) ===")
dupes = df[df.duplicated(subset=["id_code"], keep=False)]
print("Duplicate rows:", len(dupes))
print(dupes[["id_code", "diagnosis"]].head(10))
