from typing import Optional, List
from plotnine import ggplot, aes, geom_point, scale_alpha, scale_size
import pandas as pd
import numpy as np
import torch
import umap
import hdbscan
from sentence_transformers import SentenceTransformer, models
from transformers import ConvBertForMaskedLM, PreTrainedTokenizerFast


class TwitchatEmbedding:
    def __init__(self, pd_stats: pd.DataFrame):
        self.pd_stats = pd_stats
        self.vectors: np.array
        self.dimension: int

    def compute_embeddings(self):
        raise NotImplementedError


class TwitchatClustering:

    def __init__(self, embedding: TwitchatEmbedding):
        self.embedding = embedding
        self.clusters_labels: Optional[pd.Series] = None

    def compute_labels(self):
        raise NotImplementedError

    def add_time_feature(self, scale: int = 20) -> np.array:
        self.embedding.pd_stats['diff_published_datetime'] = (self.embedding.pd_stats.published_datetime - self.embedding.pd_stats.published_datetime.min()).dt.total_seconds()
        # diff_published_time_vector = ((2 * self.embedding.pd_stats['diff_published_datetime'] / self.embedding.pd_stats['diff_published_datetime'].max() - 1) * scale).tolist()
        diff_published_time_vector = (self.embedding.pd_stats['diff_published_datetime'] / scale).tolist()
        embeddig_vectors = self.embedding.vectors[:, 0:self.embedding.dimension]
        embeddig_vectors = np.concatenate((embeddig_vectors, np.transpose([diff_published_time_vector])), axis=1)
        # self.embedding.vectors = np.concatenate((self.embedding.vectors, np.transpose([diff_published_time_vector])), axis=1)
        # self.embedding.vectors = np.concatenate((self.embedding.vectors, np.transpose([diff_published_time_vector])), axis=1)
        # self.embedding.vectors = np.concatenate((self.embedding.vectors, np.transpose([diff_published_time_vector])), axis=1)
        # self.embedding.vectors = np.concatenate((self.embedding.vectors, np.transpose([diff_published_time_vector])), axis=1)
        self.embedding.vectors = embeddig_vectors
        return self.embedding.vectors

    def _add_clusters_to_df(self, clusters_labels: Optional[pd.Series] = None):
        if isinstance(clusters_labels, pd.Series):
            self.cluster_labels = clusters_labels
        self.embedding.pd_stats['labels'] = self.clusters_labels

    def plot_labels_over_time(self):
        return ggplot(self.embedding.pd_stats, aes('published_datetime', 'labels', color='labels')) + geom_point(size=0.5, alpha=.4)

    def plot_labels_x(self, labels_: List[int]):
        tmp = self.embedding.pd_stats.copy()
        tmp['labels_x'] = tmp['labels']
        tmp['alpha'] = 0.8
        tmp['size'] = 1
        tmp.loc[~tmp.labels.isin(labels_), 'labels_x'] = 0
        tmp.loc[tmp.labels_x == 0, 'alpha'] = 0.4
        tmp.loc[tmp.labels_x == 0, 'size'] = 0.5
        return ggplot(tmp, aes('x', 'y', color='factor(labels_x)', alpha='alpha', size='size')) + geom_point() + scale_alpha(guide=False) + scale_size(guide=False)

    def plot_labels_x_overt_time(self, labels_: List[int]):
        tmp = self.embedding.pd_stats.copy()
        tmp['labels_x'] = tmp['labels']
        tmp['alpha'] = 0.8
        tmp.loc[~tmp.labels.isin(labels_), 'labels_x'] = 0
        tmp.loc[tmp.labels_x == 0, 'alpha'] = 0.4
        return ggplot(tmp, aes('published_datetime', 'labels', color='factor(labels_x)', alpha='alpha')) + geom_point(size=0.5) + scale_alpha(guide=False) + scale_size(guide=False)

    def count_clusters(self):
        return self.embedding.pd_stats.labels.nunique()

    def describe_clusters(self):
        pd_desc = self.embedding.pd_stats.groupby('labels').agg({
            'message_tokenized_length': ['sum', 'mean', 'std'],
            'published_datetime': ['min', 'max'],
            'count_mention': 'sum',
            'count_url': 'sum',
            'channel': 'nunique',
            'username': 'count'
        }).rename(columns={'username': 'count_messages'}).sort_values(('count_messages', 'count')).reset_index()
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

    def plot_coords(self, with_index=False):
        if with_index:
            return ggplot(self.embedding.pd_stats, aes('x', 'y', color='index')) + geom_point(size=.5, alpha=.4)
        if self.has_labels:
            return ggplot(self.embedding.pd_stats, aes('x', 'y', color='labels')) + geom_point(size=1)
        else:
            return ggplot(self.embedding.pd_stats, aes('x', 'y')) + geom_point(size=1)

    def plot_x_over_time(self, x: str = 'x'):
        return ggplot(self.embedding.pd_stats, aes('published_datetime', x)) + geom_point(size=0.5, alpha=.4)


class TwitchatSbertEmbedding(TwitchatEmbedding):

    def __init__(self, pd_stats: pd.DataFrame, model_path_or_name: str):
        super().__init__(pd_stats)
        word_embedding_model = models.Transformer(model_path_or_name)
        pooling_model = models.Pooling(
            word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
            pooling_mode='cls'
        )
        self.model: SentenceTransformer = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.dimension = self.model.get_sentence_embedding_dimension()

    def compute_embeddings(self):
        self.vectors = self.model.encode(self.pd_stats.message_clean.to_list(), batch_size=128, show_progress_bar=True, convert_to_tensor=False)


class TwitchatSimCSEEmbedding(TwitchatEmbedding):
    def __init__(self, pd_stats: pd.DataFrame, model_path_or_name: str):
        super().__init__(pd_stats)
        self.model: SentenceTransformer = SentenceTransformer(model_path_or_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def compute_embeddings(self):
        self.vectors = self.model.encode(self.pd_stats.message_clean.to_list(), batch_size=128, show_progress_bar=True, convert_to_tensor=False)


class TwitchatPoolingEmbedding(TwitchatEmbedding):

    def __init__(self, pd_stats: pd.DataFrame, tokenizer_path: str, model_path: str):
        self.tokenizer_fast = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.model = ConvBertForMaskedLM.from_pretrained(model_path)

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
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class TwitchatUmapRepresentation(TwitchatRepresentation):

    def compute_coords(self, **umap_kwargs):
        umap_kwargs.setdefault('n_neighbors', 15)
        umap_kwargs.setdefault('n_components', 2)
        umap_kwargs.setdefault('min_dist', 0.05)
        umap_kwargs.setdefault('metric', 'cosine')
        umap_data = umap.UMAP(**umap_kwargs).fit_transform(self.embedding.vectors)
        self.pd_coords = pd.DataFrame(umap_data, columns=['x', 'y'])
        self._add_coords_to_df()


class TwitchatUmapRepresentationWithTime(TwitchatRepresentation):

    def __init__(self, embedding: TwitchatEmbedding, has_labels: bool = False):
        TwitchatRepresentation.__init__(self, embedding, has_labels)
        self.umap_1: np.array
        self.umap_2: np.array

    def compute_coords(self, umap_1_kwargs: dict = {}, umap_2_kwargs: dict = {}):
        for umap_kwargs in [umap_1_kwargs, umap_2_kwargs]:
            umap_kwargs.setdefault('n_neighbors', 15)
            umap_kwargs.setdefault('n_components', 2)
            umap_kwargs.setdefault('min_dist', 0.05)
            umap_kwargs.setdefault('metric', 'cosine')
        embedding_vectors = self.embedding.vectors[:, :self.embedding.dimension]
        datetime_vectors = self.embedding.vectors[:, self.embedding.dimension:]
        umap_data_step_1 = umap.UMAP(**umap_1_kwargs).fit_transform(embedding_vectors)
        self.vectors_after_umap_1 = umap_data_step_1
        self.vectors_before_umap_2 = np.concatenate((self.vectors_after_umap_1, datetime_vectors), axis=1)
        umap_data_step_2 = umap.UMAP(**umap_2_kwargs).fit_transform(self.vectors_before_umap_2)
        # umap_data_step_2 = umap.UMAP(**umap_2_kwargs).fit_transform(datetime_vectors)
        self.vectors_after_umap_2 = umap_data_step_2
        self.pd_coords = pd.DataFrame(self.vectors_after_umap_2, columns=['x', 'y'])
        self._add_coords_to_df()


class TwitchatUmapClustering(TwitchatClustering, TwitchatUmapRepresentationWithTime):

    def __init__(self, embedding: TwitchatEmbedding):
        TwitchatClustering.__init__(self, embedding)
        TwitchatUmapRepresentationWithTime.__init__(self, embedding, has_labels=True)
        # self.compute_coords()

    def compute_labels(self, n_components: int = 100):
        # umap_embeddings = umap.UMAP(n_neighbors=10, n_components=n_components, min_dist=0.01, metric='cosine').fit_transform(self.embedding.vectors)
        # umap_embeddings = self.embedding.pd_stats[['x', 'y']].to_numpy()
        umap_embeddings = self.vectors_before_umap_2
        # umap_embeddings = self.vectors_after_umap_1
        cluster = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.5,
                                  metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
        self.clusters_labels = pd.Series(cluster.labels_)
        self._add_clusters_to_df()
