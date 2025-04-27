import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
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

som_shape = (10, 10)
som = MiniSom(som_shape[0], som_shape[1], X_processed.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_processed)
som.train_random(X_processed, 1000)

activation_map = np.zeros(som_shape)
for x in X_processed:
    i, j = som.winner(x)
    activation_map[i, j] += 1

plt.figure(figsize=(10, 8))
plt.imshow(activation_map.T, cmap="coolwarm", origin="lower")
plt.colorbar(label="Tần suất neuron kích hoạt")
plt.title("Self-Organizing Map - Phân cụm học sinh")
plt.xlabel("Neuron hàng")
plt.ylabel("Neuron cột")
plt.grid(True)
plt.show()

cluster_labels = [som.winner(x) for x in X_processed]
df["SOM_Cluster"] = [f"{i}-{j}" for (i, j) in cluster_labels]

ic(df[["Họ và tên", "SOM_Cluster"]])

# df.to_excel("HocSinh_SOM_Clusters.xlsx", index=False)
