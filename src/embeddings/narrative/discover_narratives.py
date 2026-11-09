"""
Module: discover_narratives.py
Dự án: NarrativeCast
Mô tả: Nhận embedding và metadata bài đăng, phân cụm narrative, trích xuất bài đăng tiêu biểu
       và trực quan hóa kết quả ra file hình ảnh reports/figures/narrative_clusters.png.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min


class NarrativeDiscoverer:
    """
    Lớp xử lý khám phá narrative từ văn bản / vector embeddings.
    """
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Parameters:
        -----------
        n_clusters : int, default=4
            Số lượng cụm narrative cần khám phá.
        random_state : int, default=42
            Seed cố định cho thuật toán phân cụm.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pca = PCA(n_components=2, random_state=random_state)

    def fit_predict(
        self, 
        metadata: pd.DataFrame, 
        embeddings: Optional[np.ndarray] = None,
        text_column: str = 'content'
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, Any]]:
        """
        Đầu vào: Metadata (DataFrame) và Embeddings (tùy chọn).
        Đầu ra: DataFrame đã gán cluster_id, mảng embeddings, và danh sách bài đăng đại diện.
        """
        df = metadata.copy()
        
        # 1. Tạo embedding từ text nếu chưa truyền embeddings trực tiếp
        if embeddings is None:
            if text_column not in df.columns:
                raise ValueError(f"Không tìm thấy cột '{text_column}' trong dữ liệu metadata.")
            # Xử lý giá trị rỗng trong nội dung
            texts = df[text_column].fillna("").astype(str).tolist()
            embeddings = self.vectorizer.fit_transform(texts).toarray()

        if len(embeddings) != len(df):
            raise ValueError(f"Kích thước embeddings ({len(embeddings)}) không khớp với metadata ({len(df)}).")

        # 2. Phân cụm KMeans
        cluster_labels = self.model.fit_predict(embeddings)
        df['cluster_id'] = cluster_labels

        # 3. Tìm bài đăng tiêu biểu (gần tâm cụm nhất)
        closest_indices, _ = pairwise_distances_argmin_min(self.model.cluster_centers_, embeddings)
        representative_posts = {}
        for cluster_id, idx in enumerate(closest_indices):
            representative_posts[cluster_id] = {
                'post_id': df.iloc[idx].get('post_id', idx),
                'content': df.iloc[idx].get(text_column, ''),
                'platform': df.iloc[idx].get('platform', 'N/A')
            }

        return df, embeddings, representative_posts

    def plot_and_save_clusters(
        self, 
        embeddings: np.ndarray, 
        df_result: pd.DataFrame, 
        output_path: str = "reports/figures/narrative_clusters.png"
    ) -> None:
        """
        Vẽ biểu đồ phân bố cụm (PCA 2D) và biểu đồ số lượng bài đăng theo cụm.
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Giảm chiều xuống 2D bằng PCA
        coords_2d = self.pca.fit_transform(embeddings)
        df_plot = df_result.copy()
        df_plot['pca_x'] = coords_2d[:, 0]
        df_plot['pca_y'] = coords_2d[:, 1]

        # Tạo khung vẽ 2 đồ thị
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Đồ thị 1: Trực quan hóa không gian ngữ nghĩa (PCA Scatter plot)
        sns.scatterplot(
            data=df_plot,
            x='pca_x',
            y='pca_y',
            hue='cluster_id',
            palette='Set2',
            style='cluster_id',
            s=90,
            alpha=0.85,
            ax=axes[0]
        )
        axes[0].set_title("Không gian ngữ nghĩa phân cụm Narrative (PCA 2D)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Component 1")
        axes[0].set_ylabel("Component 2")
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Đồ thị 2: Thống kê số lượng bài đăng theo từng Cụm
        counts = df_plot['cluster_id'].value_counts().sort_index().reset_index()
        counts.columns = ['cluster_id', 'post_count']
        
        sns.barplot(
            data=counts,
            x='cluster_id',
            y='post_count',
            palette='Set2',
            ax=axes[1]
        )
        axes[1].set_title("Số lượng bài đăng trong mỗi Cụm Narrative", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("ID Cụm (Cluster ID)")
        axes[1].set_ylabel("Số lượng bài đăng")
        for p in axes[1].patches:
            axes[1].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Đã tạo và lưu hình ảnh trực quan hóa tại: {output_path}")


def discover_narratives(
    metadata: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    n_clusters: int = 4,
    fig_output_path: str = "reports/figures/narrative_clusters.png"
) -> pd.DataFrame:
    """
    Hàm đóng gói chính (Wrapper function) cho phép các module sau gọi lại.
    
    Đầu vào:
        - metadata (pd.DataFrame): Bảng dữ liệu chứa nội dung bài đăng.
        - embeddings (np.ndarray, optional): Mảng vector embedding.
        - n_clusters (int): Số lượng cụm phân tích.
        - fig_output_path (str): Đường dẫn xuất file ảnh.
        
    Đầu ra:
        - pd.DataFrame: Kết quả gán ID cụm cho từng bài đăng.
    """
    discoverer = NarrativeDiscoverer(n_clusters=n_clusters)
    df_result, emb_matrix, rep_posts = discoverer.fit_predict(metadata=metadata, embeddings=embeddings)
    discoverer.plot_and_save_clusters(emb_matrix, df_result, output_path=fig_output_path)
    
    print("\n--- TỔNG QUAN CÁC CỤM NARRATIVE KHÁM PHÁ ĐƯỢC ---")
    cluster_counts = df_result['cluster_id'].value_counts().sort_index()
    for c_id, count in cluster_counts.items():
        rep_text = rep_posts[c_id]['content'][:80].replace('\n', ' ')
        print(f"Cụm {c_id}: {count} bài đăng | Bài tiêu biểu: \"{rep_text}...\"")
        
    return df_result


# Khối thực thi trực tiếp trên tập mẫu posts_sample.csv của dự án
if __name__ == "__main__":
    sample_path = "data/samples/posts_sample.csv"
    
    if os.path.exists(sample_path):
        print(f"Đang đọc dữ liệu mẫu từ: {sample_path}")
        df_sample = pd.read_csv(sample_path)
        
        # Gọi module thực thi khám phá narrative
        res_df = discover_narratives(
            metadata=df_sample,
            n_clusters=4,
            fig_output_path="reports/figures/narrative_clusters.png"
        )
    else:
        print(f"⚠️ Không tìm thấy file mẫu tại {sample_path}. Tạo dữ liệu giả lập để kiểm thử module...")
        dummy_df = pd.DataFrame({
            'post_id': [f"P_{i:03d}" for i in range(40)],
            'content': [f"Thảo luận về chủ đề mạng xã hội và công nghệ số {i % 4}" for i in range(40)],
            'platform': ['reddit', 'bluesky', 'koo', 'voat'] * 10
        })
        res_df = discover_narratives(metadata=dummy_df, n_clusters=4)