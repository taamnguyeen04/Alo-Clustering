import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import Dataset, TensorDataset

ordinal_cols = ["Học lực", "Hạnh kiểm", "Danh hiệu"]
ordinal_mappings = [
    ['không xác định','kém','yếu','trung bình','khá','giỏi'],
    ['không xác định','yếu','trung bình','khá','tốt'],
    ['không xác định','học sinh yếu','học sinh trung bình','học sinh tiên tiến','học sinh giỏi']
]

nominal_cols = ["GVCN"]
numerical_cols = ["Toán", "Lý", "Hóa", "Sinh", "Tin", "Văn", "Sử", "Địa",
                  "Ng.ngữ", "GDCD", "C.nghệ", "Điểm TK", "K", "P", "Xếp hạng", "SSL"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ord_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=ordinal_mappings))
])

nom_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("ord", ord_pipeline, ordinal_cols),
    ("nom", nom_pipeline, nominal_cols)
])

def load_and_preprocess_data(path, test_size=0.2, random_state=42, split=True):
    df = pd.read_excel(path)

    all_cols = numerical_cols + ordinal_cols + nominal_cols
    valid_cols = [col for col in all_cols if col in df.columns]
    df = df[valid_cols].copy()

    processed = preprocessor.fit_transform(df)
    tensor_data = torch.tensor(processed, dtype=torch.float32)

    if split:
        train_array, test_array = train_test_split(tensor_data, test_size=test_size, random_state=random_state)
        return TensorDataset(train_array), TensorDataset(test_array)
    else:
        return TensorDataset(tensor_data)


if __name__ == "__main__":
    train_ds, test_ds = load_and_preprocess_data(path=r"C:\Users\tam\Documents\Data\new_student\student new\median_imputed.xlsx", split=True)
    print(f"Số mẫu train: {len(train_ds)}")
    print(f"Số mẫu test: {len(test_ds)}")

    dataset = load_and_preprocess_data(path=r"C:\Users\tam\Documents\Data\new_student\student new\median_imputed.xlsx", split=False)
    print(len(dataset))
    print(dataset[0])
