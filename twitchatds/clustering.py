from typing import Optional, List
from plotnine import ggplot, aes, geom_point
import pandas as pd
import numpy as np
import torch
import umap
import hdbscan
from sentence_transformers import SentenceTransformer, models
from transformers import MobileBertModel, PreTrainedTokenizerFast



class TwitchatEmbedding:
    def __init__(self, pd_stats: pd.DataFrame):
        self.pd_stats = pd_stats
        self.vectors: np.array

    def compute_embeddings(self):
        raise NotImplementedError


class TwitchatClustering:

    def __init__(self, embedding: TwitchatEmbedding):
        self.embedding = embedding
        self.clusters_labels: Optional[pd.Series] = None

    def compute_labels(self):
        raise NotImplementedError

    def _add_clusters_to_df(self, clusters_labels: Optional[pd.Series] = None):
        if isinstance(clusters_labels, pd.Series):
            self.cluster_labels = clusters_labels
        self.embedding.pd_stats['labels'] = self.clusters_labels

    def plot_labels_over_time(self):
        return ggplot(self.embedding.pd_stats, aes('labels', 'published_datetime', color='labels')) + geom_point(size=0.1)

    def count_clusters(self):
        return self.embedding.pd_stats.labels.nunique()

    def describe_clusters(self):
        pd_desc = self.embedding.pd_stats.groupby('labels').agg({
            'message_tokenized_length': ['sum', 'mean', 'std'],
            'published_datetime': ['min', 'max'],
            'count_mention': 'sum',
            'count_url': 'sum',
            'channel': 'count'
        }).rename(columns={'channel': 'count_messages'}).sort_values(('count_messages', 'count')).reset_index()
        pd_desc['diff_time'] = (pd_desc['published_datetime']['max'] - pd_desc['published_datetime']['min']).dt.total_seconds()
        pd_desc.columns = ["_".join(a) for a in pd_desc.columns.to_flat_index()]

        return pd_desc

    def print_cluster(self, labels: List[int]):
        for l in labels:
            print(f"-------------------- LABEL {l} --------------------")
            for msg in self.embedding.pd_stats[self.embedding.pd_stats.labels == l].message_clean.tolist():
                print(msg)


class TwitchatRepresentation:

    def __init__(self, embedding: TwitchatEmbedding, has_labels: bool = False):
        self.embedding = embedding
        self.has_labels: bool = has_labels
        self.pd_coords: pd.DataFrame

    def compute_coords(self):
        raise NotImplementedError

    def _add_coords_to_df(self, pd_coords: Optional[pd.DataFrame] = None):
        if isinstance(pd_coords, pd.DataFrame):
            self.pd_coords = pd_coords
        self.embedding.pd_stats = pd.concat([self.embedding.pd_stats.drop(columns=['x', 'y'], errors='ignore'), self.pd_coords], axis=1)

    def plot_coords(self):
        if self.has_labels:
            return ggplot(self.embedding.pd_stats, aes('x', 'y', color='labels')) + geom_point(size=1)
        else:
            return ggplot(self.embedding.pd_stats, aes('x', 'y')) + geom_point(size=1)


class TwitchatSbertEmbedding(TwitchatEmbedding):

    def __init__(self, pd_stats: pd.DataFrame, model_path_or_name: str):
        word_embedding_model = models.Transformer(model_path_or_name)
        pooling_model = models.Pooling(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
            pooling_mode='cls'
        )
        self.model: SentenceTransformer = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        super().__init__(pd_stats)

    def compute_embeddings(self):
        self.vectors = self.model.encode(self.pd_stats.message_clean.to_list(), batch_size=128, show_progress_bar=True, convert_to_tensor=False)


class TwitchatPoolingEmbedding(TwitchatEmbedding):

    def __init__(self, pd_stats: pd.DataFrame, tokenizer_path: str, model_path: str):
        self.tokenizer_fast = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.model = MobileBertModel.from_pretrained(model_path)

        super().__init__(pd_stats)

    def compute_embeddings(self):
        # Tokenize sentences
        encoded_input = self.tokenizer_fast(self.pd_stats.message_clean.to_list()[:2], padding=True, truncation=True, return_tensors='pt', max_length=128)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        self.vectors = TwitchatPoolingEmbedding.mean_pooling(model_output, encoded_input['attention_mask'])

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class TwitchatUmapRepresentation(TwitchatRepresentation):

    def compute_coords(self):
        umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.05, metric='cosine').fit_transform(self.embedding.vectors)
        self.pd_coords = pd.DataFrame(umap_data, columns=['x', 'y'])
        self._add_coords_to_df()


class TwitchatUmapClustering(TwitchatClustering, TwitchatUmapRepresentation):

    def __init__(self, embedding: TwitchatEmbedding):
        TwitchatClustering.__init__(self, embedding)
        TwitchatUmapRepresentation.__init__(self, embedding, has_labels=True)
        self.compute_coords()

    def compute_labels(self, n_components: int = 100):
        umap_embeddings = umap.UMAP(n_neighbors=15, n_components=n_components, min_dist=0.01, metric='cosine').fit_transform(self.embedding.vectors)

        cluster = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
        self.clusters_labels = pd.Series(cluster.labels_)
        self._add_clusters_to_df()
