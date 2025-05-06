import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from icecream import ic
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


df = pd.read_excel(r"../data/median_imputed.xlsx")
ordinal_cols = ["Học lực", "Hạnh kiểm", "Danh hiệu"]
nominal_cols = ["GVCN"]
numerical_cols = ["Toán", "Lý", "Hóa", "Sinh", "Tin", "Văn", "Sử", "Địa", "Ng.ngữ", "GDCD", "C.nghệ", "Điểm TK", "K", "P", "Xếp hạng", "SSL"]

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

# print("Dữ liệu gốc:", X_processed.shape)
# print("Dữ liệu sau PCA:", X_reduced.shape)

dbscan = DBSCAN(eps=0.5, min_samples=12)
clusters = dbscan.fit_predict(X_reduced)

df["Cluster"] = clusters
# ic(df[["Họ và tên", "Cluster"]])

valid_mask = clusters != -1
X_valid = X_reduced[valid_mask]
labels_valid = clusters[valid_mask]

if len(set(labels_valid)) >= 2:
    silhouette = silhouette_score(X_valid, labels_valid)
    calinski = calinski_harabasz_score(X_valid, labels_valid)
    davies = davies_bouldin_score(X_valid, labels_valid)

    print(f"{silhouette:.4f}")
    print(f"{calinski:.2f}")
    print(f"{davies:.4f}")

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_processed)

# --- Vẽ biểu đồ PCA ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Clustering by DBSCAN Model (PCA Reduced)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)

# --- Vẽ biểu đồ t-SNE ---
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Clustering by DBSCAN Model (t-SNE Reduced)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)

plt.colorbar(label='Cluster', ax=plt.gca(), shrink=0.75)
plt.tight_layout()
plt.show()

