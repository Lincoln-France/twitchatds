from typing import Optional, List
from plotnine import ggplot, aes, geom_point, scale_alpha, scale_size
import re
import pandas as pd
import numpy as np
import torch
import umap
import hdbscan
from sentence_transformers import SentenceTransformer, models
from transformers import ConvBertForMaskedLM, PreTrainedTokenizerFast
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_samples
from twitchatds.berttopic import MyBERTopic
from bertopic import BERTopic
import plotly.express as px


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
        self.clustering_model = None
        self.pd_stats_prepared = None

    def compute_labels(self):
        raise NotImplementedError

    def add_time_feature(self, scale: int = 20) -> np.array:
        self.embedding.pd_stats['diff_published_datetime'] = (self.embedding.pd_stats.published_datetime - self.embedding.pd_stats.published_datetime.min()).dt.total_seconds()
        diff_published_time_vector = (self.embedding.pd_stats['diff_published_datetime'] / scale).tolist()
        embeddig_vectors = self.embedding.vectors[:, 0:self.embedding.dimension]
        embeddig_vectors = np.concatenate((embeddig_vectors, np.transpose([diff_published_time_vector])), axis=1)
        self.embedding.vectors = embeddig_vectors
        return self.embedding.vectors

    def _add_clusters_to_df(self, clusters_labels: Optional[pd.Series] = None):
        if isinstance(clusters_labels, pd.Series):
            self.clusters_labels = clusters_labels
        self.embedding.pd_stats['labels'] = self.clusters_labels

    def _prepare_df(self, labels_: List[int] = [], force=False):
        if self.pd_stats_prepared is None or force:
            tmp = self.embedding.pd_stats.copy()
            # create silhouette
            tmp['silhouette'] = silhouette_samples(self.embedding.vectors, tmp.labels)
            # create order of labels
            order_labels = tmp.sort_values('published_datetime', ascending=True).groupby('labels').first().sort_values('published_datetime', ascending=True).reset_index()
            order_labels['order'] = order_labels.index.to_series()
            order_labels = order_labels[['order', 'labels']]
            tmp = pd.merge(tmp, order_labels, on='labels')
            # elapsed_time par cluster
            tmp['elapsed_time_label'] = tmp.groupby('labels').published_datetime.transform(lambda x: (x - x.min()).dt.total_seconds())
            # set property
            self.pd_stats_prepared = tmp

        if labels_:
            return self.pd_stats_prepared[self.pd_stats_prepared.labels.isin(labels_)]

        return self.pd_stats_prepared

    def plot_labels_over_time(self):
        tmp = self._prepare_df()
        tmp = tmp[tmp.labels != -1]
        return ggplot(tmp, aes('published_datetime', 'order', color='labels')) + geom_point(size=0.5, alpha=.4)

    def plot_labels_x(self, labels_: List[int]):
        tmp = self.embedding.pd_stats.copy()
        tmp['labels_x'] = tmp['labels']
        tmp['alpha'] = 0.8
        tmp['size'] = 1
        tmp.loc[~tmp.labels.isin(labels_), 'labels_x'] = -2
        tmp.loc[tmp.labels_x == -2, 'alpha'] = 0.4
        tmp.loc[tmp.labels_x == -2, 'size'] = 0.5
        return ggplot(tmp, aes('x', 'y', color='factor(labels_x)', alpha='alpha', size='size')) + geom_point() + scale_alpha(guide=False) + scale_size(guide=False)

    def plot_labels_x_over_time(self, labels_: List[int], include_all_labels=True):
        if isinstance(include_all_labels, list):
            tmp = self._prepare_df(labels_=include_all_labels)
        else:
            tmp = self._prepare_df()
        tmp['labels_x'] = tmp['labels']
        tmp['alpha'] = 0.8
        tmp.loc[~tmp.labels.isin(labels_), 'labels_x'] = 0
        tmp.loc[tmp.labels_x == 0, 'alpha'] = 0.4

        return ggplot(tmp, aes('published_datetime', 'order', color='factor(labels_x)', alpha='alpha')) + geom_point(size=0.5) + scale_alpha(guide=False) + scale_size(guide=False)

    def plot_mean_labels(self, labels_: List[int] = []):
        tmp = self.embedding.pd_stats.copy()
        tmp_g = tmp.groupby('labels').agg({'x': 'mean', 'y': 'mean'}).reset_index()
        tmp_g['labels_x'] = tmp_g['labels']
        tmp_g['alpha'] = 0.8
        tmp_g.loc[~tmp_g.labels.isin(labels_), 'labels_x'] = 0
        tmp_g.loc[tmp_g.labels_x == 0, 'alpha'] = 0.4
        return ggplot(tmp_g, aes('x', 'y', color='factor(labels_x)', alpha='alpha')) + geom_point(size=1) + scale_alpha(guide=False) + scale_size(guide=False)

    def plot_silhouette_by_cluster(self, labels_: List[int] = []):
        tmp = self._prepare_df()
        tmp = tmp[tmp.labels != -1]

        pd_data = tmp.groupby('labels').agg({'silhouette': 'mean'}).reset_index()

        pd_data

        pass

    def count_clusters(self):
        return self.embedding.pd_stats.labels.nunique()

    def count_message_without_cluster(self):
        pd_desc = self.describe_clusters()
        return pd_desc[pd_desc.labels_ == -1]['count_messages_count'].tolist()

    def describe_clusters(self, emotes: List[str]):

        tmp = self._prepare_df()

        # premier mot
        tmp['first_word'] = tmp.message.str.lower().str.split(' ').str[0]
        pd_first_word = tmp.groupby(['labels', 'first_word'])\
            .first_word.count()\
            .reset_index(name='first_word_count')\
            .sort_values('first_word_count', ascending=False)\
            .groupby(['labels'])\
            .first()\
            .reset_index()

        # endswith
        tmp['endswith_question'] = tmp.message.str.strip().str.endswith('?')
        tmp['endswith_exclamation'] = tmp.message.str.strip().str.endswith('!')

        # startswith
        tmp['startswith_command'] = tmp.message.str.strip().str.startswith('!')

        # emojis
        tmp['count_emotes'] = tmp.message.str.count('|'.join([re.escape(e) for e in emotes]))

        pd_desc = tmp.groupby('labels').agg({
            'message_tokenized_length': ['sum', 'mean', 'std'],
            'published_datetime': ['min', 'max'],
            'elapsed_time_label': ['max', 'std', 'median'],
            'count_mention': ['sum', 'mean'],
            'count_url': ['sum', 'mean'],
            'endswith_question': ['sum', 'mean'],
            'endswith_exclamation': ['sum', 'mean'],
            'startswith_command': ['sum', 'mean'],
            'count_emotes': ['sum', 'mean'],
            'channel': 'nunique',
            'index': 'count',
            'username': 'nunique',
            'silhouette': ['mean', 'std'],
            'x': 'mean',
            'y': 'mean'
        }).rename(columns={'index': 'count_messages'}).sort_values(('count_messages', 'count')).reset_index()
        pd_desc['diff_time'] = (pd_desc['published_datetime']['max'] - pd_desc['published_datetime']['min']).dt.total_seconds()
        pd_desc.columns = ["_".join(a) for a in pd_desc.columns.to_flat_index()]

        pd_desc = pd.merge(pd_desc, pd_first_word, left_on='labels_', right_on='labels')
        pd_desc.drop(columns=['labels'])

        pd_desc['first_word_mean'] = pd_desc['first_word_count'] / pd_desc['count_messages_count']

        return pd_desc

    def print_cluster(self, labels: List[int], **kwargs):
        for label in labels:
            tmp = self._prepare_df()
            pd_sel = tmp[tmp.labels == label]
            print("silouhette moyenne: {:.2f}".format(pd_sel.silhouette.mean()), **kwargs)
            print(f"-------------------- LABEL {label} --------------------", **kwargs)
            for _, row in pd_sel.iterrows():
                print("{:%Y-%m-%d %H:%M:%S} {: .2f}: {} ".format(row['published_datetime'], row['silhouette'], row['message_clean']), **kwargs)


class TwitchatRepresentation:

    def __init__(self, embedding: TwitchatEmbedding, has_labels: bool = False):
        self.embedding = embedding
        self.has_labels: bool = has_labels
        self.pd_coords: pd.DataFrame
        self.representation_model = None

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
        token_embeddings = model_output[0]
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
        self.representation_model = umap.UMAP(**umap_kwargs)
        umap_data = self.representation_model.fit_transform(self.embedding.vectors)
        self.pd_coords = pd.DataFrame(umap_data, columns=['x', 'y'])
        self._add_coords_to_df()


class TwitchatUmapRepresentationWithTime(TwitchatRepresentation):

    def __init__(self, embedding: TwitchatEmbedding, has_labels: bool = False):
        TwitchatRepresentation.__init__(self, embedding, has_labels)
        self.umap_1: np.array
        self.umap_2: np.array

    def compute_coords(self, umap_1_kwargs: dict = {}, umap_2_kwargs: dict = {}, only_first: bool = False):
        for umap_kwargs in [umap_1_kwargs, umap_2_kwargs]:
            umap_kwargs.setdefault('n_neighbors', 15)
            umap_kwargs.setdefault('n_components', 2)
            umap_kwargs.setdefault('min_dist', 0.05)
            umap_kwargs.setdefault('metric', 'cosine')
        embedding_vectors = self.embedding.vectors[:, :self.embedding.dimension]
        datetime_vectors = self.embedding.vectors[:, self.embedding.dimension:]
        self.representation_model = umap.UMAP(**umap_1_kwargs)
        umap_data_step_1 = self.representation_model.fit_transform(embedding_vectors)
        self.vectors_after_umap_1 = umap_data_step_1

        if only_first:
            assert self.vectors_after_umap_1.shape[0] == 2
            self.pd_coords = pd.DataFrame(self.vectors_after_umap_1, columns=['x', 'y'])
            self._add_coords_to_df()
            return

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
        cluster = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_epsilon=0.2,
                                  metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
        self.clusters_labels = pd.Series(cluster.labels_)
        self.model = cluster
        self._add_clusters_to_df()


def get_default_count_vectorizer(fit: bool = True, documents: List[str] = []) -> CountVectorizer:
    stop_words = stopwords.words('french') + stopwords.words('english')
    stop_words = stop_words + [w.capitalize() for w in stop_words]
    stop_words = stop_words + ['▁' + w for w in stop_words] + ['▁']
    count_vect_model = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        strip_accents='ascii',
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2)
    )

    if fit:
        count_vect_model.fit(documents)

    return count_vect_model


class TwitchatBertTopicClustering(TwitchatClustering, TwitchatUmapRepresentationWithTime):

    count_vect_model = None

    def __init__(self, embedding: TwitchatEmbedding):
        TwitchatClustering.__init__(self, embedding)
        TwitchatUmapRepresentationWithTime.__init__(self, embedding, has_labels=True)

        if self.__class__.count_vect_model is None:
            print("update class attribute count_vect_model")
            self.__class__.count_vect_model = get_default_count_vectorizer(True, self.embedding.pd_stats.message_clean.to_list())

    @staticmethod
    def sample_messages(x: list):
        return '<br>'.join([a[:75] + '...' if len(a) > 75 else a for a in x[:20]])

    @staticmethod
    def sample_topics_terms(x: list):
        return ' / '.join(x[:5])

    def compute_labels(self):

        umap_model = self.representation_model
        hdbscan_model = hdbscan.HDBSCAN(**{'min_cluster_size': 5, 'min_samples': 1, 'cluster_selection_epsilon': 0.5, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'prediction_data': True})
        self.topic_model = MyBERTopic(
            language="french",
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=self.count_vect_model,
            calculate_probabilities=True,
            verbose=True,
        )
        topics, probs = self.topic_model.fit_transform(self.embedding.pd_stats.message_clean.to_list(), self.vectors_before_umap_2.copy())
        self.documents = self.embedding.pd_stats.message_clean.to_list()
        self.topics = topics
        self.probabilities = probs
        self._add_clusters_to_df(pd.Series(topics))

        return

    def _get_topics_terms_df(self) -> pd.DataFrame:
        pd_topic_terms_list = []
        for current_label, terms in self.topic_model.get_topics().items():
            pd_tmp = pd.DataFrame(terms, columns=['terms', 'importance'])
            pd_tmp['labels'] = current_label
            pd_topic_terms_list.append(pd_tmp)

        pd_topic_terms = pd.concat(pd_topic_terms_list)
        pd_topic_terms.sort_values(['labels', 'importance'], ascending=False)

        return pd_topic_terms

    def _get_terms_description(self) -> pd.DataFrame:

        colnames_term = ['term_{}'.format(i + 1) for i in range(10)]
        colnames_term_freq = ['term_{}_freq'.format(i + 1) for i in range(10)]
        colnames_final = ['labels_', 'terms_nunique']
        for term, term_freq in zip(colnames_term, colnames_term_freq):
            colnames_final.append(term)
            colnames_final.append(term_freq)

        pd_topics_terms = self._get_topics_terms_df()
        pd_topics_terms['row_number'] = pd_topics_terms.groupby('labels').cumcount()

        pd_terms_value = pd_topics_terms.pivot_table(columns='row_number', index='labels', values='terms', aggfunc=' '.join, fill_value='')
        pd_terms_value.columns = colnames_term

        pd_terms_freq = pd_topics_terms.pivot_table(columns='row_number', index='labels', values='importance', fill_value=0, aggfunc=int)
        pd_terms_freq.columns = colnames_term_freq

        pd_terms = pd.merge(pd_terms_value, pd_terms_freq, on='labels')
        pd_terms = pd_terms.reset_index()
        pd_terms['terms_nunique'] = pd.Series(np.count_nonzero(self.topic_model.c_tf_idf.toarray(), axis=1))

        pd_terms = pd_terms.rename(columns={'labels': 'labels_'})
        pd_terms = pd_terms[colnames_final]

        return pd_terms

    def describe_clusters(self, emotes: List[str]) -> pd.DataFrame:
        desc_part_1 = super().describe_clusters(emotes=emotes)
        desc_part_2 = self._get_terms_description()

        pd_desc = pd.merge(desc_part_1, desc_part_2, on='labels_')
        pd_desc['terms_nunique_mean'] = pd_desc['terms_nunique'] / pd_desc['count_messages_count']

        return pd_desc

    def print_cluster(self, labels: List[int], **kwargs):
        for label in labels:
            if label == -1:
                continue
            print("-------------------- LABEL %s --------------------" % label, **kwargs)
            topn_words = self.topic_model.get_topic(label)
            print('/'.join(['%s (%s)' % (k, np.round(v, 2)) for k, v in topn_words[:5]]), **kwargs)
            super().print_cluster([label], **kwargs)

    def plotly_labels_x(self, labels_: List[int] = []):
        tmp = self.embedding.pd_stats.copy()
        tmp['labels_x'] = tmp['labels']
        tmp['alpha'] = 0.8
        tmp['size'] = 1
        tmp.loc[~tmp.labels.isin(labels_), 'labels_x'] = -2
        tmp.loc[tmp.labels_x == -2, 'alpha'] = 0.4
        tmp.loc[tmp.labels_x == -2, 'size'] = 0.5

        fig = px.scatter(tmp, x='x', y='y', color='labels_x', hover_data=['message_clean'])

        return fig

    def plotly_mean_labels(self, labels_: List[int] = []):

        pd_topic_terms = self._get_topics_terms_df()

        pd_topic_terms_g = pd_topic_terms.groupby('labels').agg({'terms': self.sample_topics_terms}).reset_index()

        tmp = self.embedding.pd_stats.copy()

        tmp_g = tmp.groupby('labels').agg({'x': 'mean', 'y': 'mean', 'message_clean': self.sample_messages}).reset_index()
        tmp_g['labels_x'] = tmp_g['labels']
        tmp_g['color'] = 'important'
        tmp_g.loc[~tmp_g.labels.isin(labels_), 'labels_x'] = 0
        tmp_g.loc[tmp_g.labels_x == 0, 'color'] = 'not important'

        tmp_g = pd.merge(tmp_g, pd_topic_terms_g, how='left', on='labels')

        fig = px.scatter(tmp_g, x='x', y='y', color='color', hover_data={
            'color': False,
            'message_clean': True,
            'x': False,
            'y': False,
            'labels': True,
            'terms': True
        })
        hovertemplate = '<b>%{customdata[5]}</b><br>Label %{customdata[4]}<br>%{customdata[1]}'
        fig.update_traces(selector={'name': 'not important'}, mode='markers', marker=dict(size=10), opacity=0.5, hovertemplate=hovertemplate)
        fig.update_traces(
            selector={'name': 'important'},
            mode='markers',
            marker=dict(size=10, color='LightSkyBlue', line=dict(color='MediumPurple', width=2)),
            opacity=0.8,
            hovertemplate=hovertemplate
        )

        return fig

    def plotly_labels_x_over_time(self, labels_: List[int], include_all_labels=True):
        if isinstance(include_all_labels, list):
            tmp = self._prepare_df(labels_=include_all_labels)
        else:
            tmp = self._prepare_df()
        tmp['labels_x'] = tmp['labels']
        # tmp['selection'] = 'important'
        tmp.loc[~tmp.labels.isin(labels_), 'labels_x'] = 0
        # tmp.loc[tmp.labels_x == 0, 'selection'] = 'not important'

        pd_topic_terms = self._get_topics_terms_df()
        pd_topic_terms_g = pd_topic_terms.groupby('labels').agg({'terms': self.sample_topics_terms}).reset_index()
        pd_topic_terms_g['selection'] = pd_topic_terms_g['labels'].astype(str) + '_' + pd_topic_terms_g['terms']

        tmp = pd.merge(tmp, pd_topic_terms_g, how='inner', on='labels')

        fig = px.line(tmp, x='published_datetime', y='order', color='selection', hover_data=['labels', 'message_clean'])
        fig.update_traces(mode='markers+lines', marker=dict(size=4), line=dict(width=1))

        # fig = px.scatter(tmp, x='published_datetime', y='order', color='selection', hover_data=['labels', 'message_clean'])
        # fig.update_traces(selector={'name': 'not important'}, mode='markers', marker=dict(size=1), opacity=0.5)
        # fig.update_traces(
        #     selector={'name': 'important'},
        #     mode='markers',
        #     marker=dict(size=3, color='LightSkyBlue', line=dict(color='MediumPurple', width=1)),
        #     opacity=1
        # )
        # fig.update_traces(orientation='h', side='positive', width=3, points=False)

        return fig

    def plotly_mean_labels_over_time(self, labels_: List[int] = []):

        pd_topic_terms = self._get_topics_terms_df()

        pd_topic_terms_g = pd_topic_terms.groupby('labels').agg({'terms': self.sample_topics_terms}).reset_index()

        tmp = self._prepare_df().copy()

        tmp_g = tmp.groupby('labels').agg({'x': 'mean', 'y': 'mean', 'published_datetime': 'mean', 'order': 'mean', 'message_clean': self.sample_messages}).reset_index()
        tmp_g['labels_x'] = tmp_g['labels']
        tmp_g['color'] = 'important'
        tmp_g.loc[~tmp_g.labels.isin(labels_), 'labels_x'] = 0
        tmp_g.loc[tmp_g.labels_x == 0, 'color'] = 'not important'

        tmp_g = pd.merge(tmp_g, pd_topic_terms_g, how='left', on='labels')

        fig = px.scatter(tmp_g, x='published_datetime', y='order', color='color', hover_data={
            'color': False,
            'message_clean': True,
            'x': False,
            'y': False,
            'labels': True,
            'terms': True
        })
        hovertemplate = '<b>%{customdata[5]}</b><br>Label %{customdata[4]}<br>%{customdata[1]}'
        fig.update_traces(selector={'name': 'not important'}, mode='markers', marker=dict(size=10), opacity=0.5, hovertemplate=hovertemplate)
        fig.update_traces(
            selector={'name': 'important'},
            mode='markers',
            marker=dict(size=10, color='LightSkyBlue', line=dict(color='MediumPurple', width=2)),
            opacity=0.8,
            hovertemplate=hovertemplate
        )

        return fig


class TwitchatSimpleBertTopicClustering(TwitchatBertTopicClustering):

    def compute_labels(self):

        self.topic_model = BERTopic(
            language="french",
            embedding_model=self.embedding.model,
            # umap_model=umap_model,
            # hdbscan_model=hdbscan_model,
            vectorizer_model=self.count_vect_model,
            calculate_probabilities=True,
            verbose=True,
        )
        topics, probs = self.topic_model.fit_transform(self.embedding.pd_stats.message_clean.to_list())
        self.documents = self.embedding.pd_stats.message_clean.to_list()
        self.topics = topics
        self.probabilities = probs
        self._add_clusters_to_df(pd.Series(topics))

        return
