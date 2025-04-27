import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model import Encoder, ClusteringHead
from dataset import load_and_preprocess_data
from icecream import ic
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def save_checkpoint(path, epoch, encoder, clustering_head, optimizer):
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'clustering_head_state_dict': clustering_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def save_best_checkpoint(path, encoder, clustering_head):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'clustering_head_state_dict': clustering_head.state_dict(),
    }, path)

def load_checkpoint(path, encoder, clustering_head, optimizer, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        clustering_head.load_state_dict(checkpoint['clustering_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1
    return 0

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity_matrix.masked_fill_(mask, -1e9)

    positive_sim = torch.sum(z1 * z2, dim=1)
    positives = torch.cat([positive_sim, positive_sim], dim=0)

    logits = similarity_matrix / temperature
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    loss = F.cross_entropy(logits, labels)
    return loss

def clustering_loss(cluster_logits):
    p = F.softmax(cluster_logits, dim=1)
    q = (p ** 2) / torch.sum(p, dim=0)
    q = q / torch.sum(q, dim=1, keepdim=True)

    loss = F.kl_div(p.log(), q.detach(), reduction='batchmean')
    return loss

def cluster_compactness_loss(embeddings, cluster_assignments, num_clusters):
    cluster_centers = []
    loss = 0.0
    for k in range(num_clusters):
        cluster_points = embeddings[cluster_assignments == k]
        if len(cluster_points) > 0:
            center = cluster_points.mean(dim=0)
            cluster_centers.append(center)
            loss += ((cluster_points - center) ** 2).sum()
    loss = loss / embeddings.size(0)
    return loss, cluster_centers

def cluster_separation_loss(cluster_centers):
    loss = 0.0
    num_centers = len(cluster_centers)
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            dist = (cluster_centers[i] - cluster_centers[j]).pow(2).sum()
            loss += 1.0 / (dist + 1e-6)
    return loss

def total_loss_fn(z1, z2, cluster_logits, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    loss_contrastive = contrastive_loss(z1, z2)
    loss_cluster = clustering_loss(cluster_logits)

    batch_size = z1.size(0)
    cluster_logits_z1 = cluster_logits[:batch_size]
    cluster_assignments = cluster_logits_z1.argmax(dim=1)

    compact_loss, cluster_centers = cluster_compactness_loss(z1, cluster_assignments, num_clusters=2)
    separation_loss = cluster_separation_loss(cluster_centers)

    total = alpha * loss_contrastive + beta * loss_cluster + gamma * compact_loss + delta * separation_loss
    return total

def augment(x, noise_scale=0.1, dropout_prob=0.1, scale_jitter=0.05):
    if noise_scale > 0:
        noise = noise_scale * torch.randn_like(x)
        x = x + noise
    if dropout_prob > 0:
        mask = (torch.rand_like(x) > dropout_prob).float()
        x = x * mask
    if scale_jitter > 0:
        scale = 1.0 + scale_jitter * (2 * torch.rand_like(x) - 1)
        x = x * scale
    return x

def plot_clusters(X_reduced, clusters, epoch, out_path="out"):
    plt.figure(figsize=(8,6))
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=clusters, cmap='viridis', s=10)
    plt.title(f'Clustering at Epoch {epoch}')
    plt.savefig(os.path.join(out_path, f"clustering_epoch_{epoch}.png"))
    plt.close()

def evaluate(encoder, clustering_head, test_dataloader, device):
    encoder.eval()
    clustering_head.eval()

    all_test_data = torch.cat([x[0] for x in test_dataloader], dim=0).to(device)

    with torch.no_grad():
        embeddings = encoder(all_test_data)
        cluster_logits = clustering_head(embeddings)
        clusters = cluster_logits.argmax(dim=1).cpu().numpy()
        X_embedded = embeddings.cpu().numpy()

    X_reduced = PCA(n_components=2).fit_transform(X_embedded)

    silhouette = silhouette_score(X_reduced, clusters)
    calinski = calinski_harabasz_score(X_reduced, clusters)
    davies = davies_bouldin_score(X_reduced, clusters)

    print(f"Test set | Silhouette: {silhouette:.4f} | Calinski: {calinski:.2f} | Davies: {davies:.4f}")

def train():
    batch_size = 128
    lr = 1e-5
    num_epochs = 50
    embedding_dim = 64
    out_path = "out_knn_imputed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_path, "logs"))

    train_dataset, test_dataset = load_and_preprocess_data(path=r"data/knn_imputed.xlsx", split=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    encoder = Encoder(input_dim=20, embedding_dim=embedding_dim).to(device)
    clustering_head = ClusteringHead(embedding_dim=embedding_dim, num_clusters=2).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(clustering_head.parameters()), lr=lr)

    start_epoch = load_checkpoint(os.path.join(out_path, 'checkpoint.pt'), encoder, clustering_head, optimizer, device)

    best_silhouette = -1
    patience = 5
    wait = 0
    noise_scale = 0.1

    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        clustering_head.train()
        total_loss_val = 0

        for (x,) in train_dataloader:
            x = x.to(device)
            x1 = augment(x, noise_scale=noise_scale)
            x2 = augment(x, noise_scale=noise_scale)

            z1 = encoder(x1)
            z2 = encoder(x2)
            z_all = torch.cat([z1, z2], dim=0)
            cluster_logits = clustering_head(z_all)

            loss = total_loss_fn(z1, z2, cluster_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss_val / len(train_dataloader):.4f}")

        save_checkpoint(os.path.join(out_path, 'checkpoint.pt'), epoch, encoder, clustering_head, optimizer)

        encoder.eval()
        clustering_head.eval()

        all_train_data = torch.cat([x[0] for x in train_dataloader], dim=0).to(device)

        with torch.no_grad():
            embeddings = encoder(all_train_data)
            cluster_logits = clustering_head(embeddings)
            clusters = cluster_logits.argmax(dim=1).cpu().numpy()
            X_embedded = embeddings.cpu().numpy()

        X_reduced = PCA(n_components=2).fit_transform(X_embedded)

        silhouette = silhouette_score(X_reduced, clusters)
        calinski = calinski_harabasz_score(X_reduced, clusters)
        davies = davies_bouldin_score(X_reduced, clusters)

        print(f"Epoch {epoch} | Silhouette: {silhouette:.4f} | Calinski: {calinski:.2f} | Davies: {davies:.4f}")

        writer.add_scalar('Metrics/Silhouette', silhouette, epoch)
        writer.add_scalar('Metrics/Calinski_Harabasz', calinski, epoch)
        writer.add_scalar('Metrics/Davies_Bouldin', davies, epoch)

        plot_clusters(X_reduced, clusters, epoch, out_path=out_path)

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            wait = 0
            save_best_checkpoint(os.path.join(out_path, 'best_model.pt'), encoder, clustering_head)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered!")
                break

        if (epoch + 1) % 5 == 0:
            noise_scale *= 0.8

    evaluate(encoder, clustering_head, test_dataloader, device)

if __name__ == '__main__':
    train()
