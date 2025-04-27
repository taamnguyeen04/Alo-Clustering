import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from icecream import ic

df = pd.read_excel(r"../data/median_imputed.xlsx")
ordinal_cols = ["Học lực", "Hạnh kiểm", "Danh hiệu"]
nominal_cols = ["GVCN"]
numerical_cols = ["Toán", "Lý", "Hóa", "Sinh", "Tin", "Văn", "Sử", "Địa", "Ng.ngữ", "GDCD", "C.nghệ", "Điểm TK", "K", "P", "SSL"]

ordinal_mappings = [
    ['không xác định', 'kém', 'yếu', 'trung bình', 'khá', 'giỏi'],
    ['không xác định', 'yếu', 'trung bình', 'khá', 'tốt'],
    ['không xác định', 'học sinh yếu', 'học sinh trung bình', 'học sinh tiên tiến', 'học sinh giỏi']
]

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

X = df[numerical_cols + ordinal_cols + nominal_cols]
X_processed = preprocessor.fit_transform(X)

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_processed)

print("Dữ liệu gốc:", X_processed.shape)
print("Dữ liệu sau PCA:", X_reduced.shape)

iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
anomaly_pred = iso_forest.fit_predict(X_reduced)

df["Anomaly"] = anomaly_pred
df["Anomaly_Label"] = df["Anomaly"].map({1: "Bình thường", -1: "Bất thường"})
# ic(df[["Họ và tên", "Anomaly_Label"]])

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=anomaly_pred, cmap="coolwarm", s=50)
plt.title("Isolation Forest – Phát hiện học sinh bất thường")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.colorbar(label="Bất thường (-1) / Bình thường (1)")
plt.show()
