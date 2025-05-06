import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from model import Encoder, ClusteringHead

# Cấu hình
MODEL_PATH = r"C:\Users\tam\Documents\GitHub\Alo-Clustering\out_mode_imputed\best_model.pt"
DATA_PATH = r"data/median_imputed.xlsx"  # hoặc sửa nếu chạy ngoài repo
EMBEDDING_DIM = 64
NUM_CLUSTERS = 2

# Tiền xử lý dữ liệu
df = pd.read_excel(DATA_PATH)
ordinal_cols = ["Học lực", "Hạnh kiểm", "Danh hiệu"]
nominal_cols = ["GVCN"]
numerical_cols = ["Toán", "Lý", "Hóa", "Sinh", "Tin", "Văn", "Sử", "Địa",
                  "Ng.ngữ", "GDCD", "C.nghệ", "Điểm TK", "K", "P", "Xếp hạng", "SSL"]
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

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_dim=X_processed.shape[1], embedding_dim=EMBEDDING_DIM).to(device)
clustering_head = ClusteringHead(embedding_dim=EMBEDDING_DIM, num_clusters=NUM_CLUSTERS).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
clustering_head.load_state_dict(checkpoint['clustering_head_state_dict'])

encoder.eval()
clustering_head.eval()

# Đưa dữ liệu vào mô hình
with torch.no_grad():
    inputs = torch.tensor(X_processed, dtype=torch.float32).to(device)
    embeddings = encoder(inputs)
    cluster_logits = clustering_head(embeddings)
    clusters = torch.argmax(cluster_logits, dim=1).cpu().numpy()

# Đánh giá định tính
X_embedded = embeddings.cpu().numpy()
X_reduced = PCA(n_components=2).fit_transform(X_embedded)

silhouette = silhouette_score(X_reduced, clusters)
calinski = calinski_harabasz_score(X_reduced, clusters)
davies = davies_bouldin_score(X_reduced, clusters)

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz Index: {calinski:.2f}")
print(f"Davies-Bouldin Index: {davies:.4f}")

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_processed)

# --- Vẽ biểu đồ PCA ---
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Clustering by Alo Model (PCA Reduced)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)

# --- Vẽ biểu đồ t-SNE ---
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Clustering by Alo Model (t-SNE Reduced)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)

plt.colorbar(label='Cluster', ax=plt.gca(), shrink=0.75)
plt.tight_layout()
plt.show()
