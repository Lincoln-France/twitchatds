from bertopic import BERTopic
from bertopic._ctfidf import ClassTFIDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_array
import scipy.sparse as sp
import numpy as np
import pandas as pd


class MyClassTFIDF(ClassTFIDF):

    def fit(self, X, n_samples: int, multiplier: np.ndarray = None):
        """Learn the idf vector (global term weights).

        Arguments:
            X: A matrix of term/token counts.
            n_samples: Number of total documents
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64
        if self.use_idf:
            _, n_features = X.shape
            df = np.squeeze(np.asarray(X.sum(axis=0)))
            avg_nr_samples = int(X.sum(axis=1).mean())
            idf = np.log(1 + avg_nr_samples / df)
            if multiplier is not None:
                idf = idf * multiplier
            self._idf_diag = sp.diags(idf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)

        return self


class MyBERTopic(BERTopic):

    def _reduce_dimensionality(self, embeddings, y):
        return np.nan_to_num(embeddings)

    def _c_tf_idf(self, documents_per_topic: pd.DataFrame, m: int, fit: bool = True):
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)
            fit: Whether to fit a new vectorizer or use the fitted self.vectorizer_model

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic.Document.values)

        # if fit:
        #     print("Fitting vectorizer model")
        #     self.vectorizer_model.fit(documents)

        words = self.vectorizer_model.get_feature_names()
        X = self.vectorizer_model.transform(documents)

        if self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([1.2 if word in seed_topic_list else 1 for word in words])
        else:
            multiplier = None

        if fit:
            tfidf_instance = MyClassTFIDF()
            tfidf_instance.use_idf = False
            self.transformer = tfidf_instance.fit(X, n_samples=m, multiplier=multiplier)

        c_tf_idf = self.transformer.transform(X)

        self.topic_sim_matrix = cosine_similarity(c_tf_idf)

        return c_tf_idf, words
