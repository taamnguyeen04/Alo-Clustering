import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from icecream import ic

df = pd.read_excel(r"../data/knn_imputed.xlsx")
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

# Tiền xử lý tổng
X = df[numerical_cols + ordinal_cols + nominal_cols]
X_processed = preprocessor.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_processed)

# print("Dữ liệu gốc:", X_processed.shape)
# print("Dữ liệu sau PCA:", X_reduced.shape)

# Clustering bằng GMM (Expectation-Maximization)
gmm = GaussianMixture(n_components=2, random_state=42)
clusters = gmm.fit_predict(X_reduced)

# Gán cụm vào DataFrame
df["Cluster"] = clusters
# ic(df[["Họ và tên", "Cluster"]])

if len(set(clusters)) >= 2:
    silhouette = silhouette_score(X_reduced, clusters)
    calinski = calinski_harabasz_score(X_reduced, clusters)
    davies = davies_bouldin_score(X_reduced, clusters)

    print(f"{silhouette:.4f}")
    print(f"{calinski:.2f}")
    print(f"{davies:.4f}")

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', s=50)
# plt.title("Expectation-Maximization (GMM) Clustering Sau PCA")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.colorbar(scatter, label='Cluster')
# plt.grid(True)
# plt.show()
