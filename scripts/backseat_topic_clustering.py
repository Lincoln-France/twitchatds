"""
Topic clustering using SimCSE
"""
# imports
import os
import pandas as pd
import numpy as np
import twitchatds
from twitchatds import get_backseat
from twitchatds.clustering import TwitchatSimCSEEmbedding, TwitchatBertTopicClustering
from tokenizers import Tokenizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main(in_file: str, tokenizer_file: str, model_path_or_name: str, date: str, limit: int, scale: int, clustering_directory: str):

    if scale is None:
        scale_directory = "no_scale"
    else:
        scale_directory = "scale_{}".format(scale)

    print("Run {}".format(scale_directory))

    # paramètres
    save_directory = os.path.join(clustering_directory, scale_directory)

    # récupération des données
    all_streams = pd.read_pickle(in_file)
    tokenizer = Tokenizer.from_file(tokenizer_file)

    # format des données
    backseat_stream = get_backseat(all_streams, date, tokenizer)

    emotes_channel_name = twitchatds.FIXED_GLOBAL_EMOTES
    emotes_global_name = twitchatds.FIXED_JEANMASSIET_EMOTES

    # limit
    if limit:
        backseat_stream = backseat_stream.head(limit)

    # chargement des modèles/tokenizers
    embedding_mlm_simcse = TwitchatSimCSEEmbedding(backseat_stream, model_path_or_name)

    # calcul de l'embedding
    embedding_mlm_simcse.compute_embeddings()

    # ajout du temps
    embedding_clu = TwitchatBertTopicClustering(embedding_mlm_simcse)
    if scale:
        embedding_clu.add_time_feature(scale=scale)

    # calcul umap
    embedding_clu.compute_coords(
        umap_1_kwargs={'n_neighbors': 100, 'min_dist': 0.99, 'metric': 'cosine', 'n_components': 10, 'random_state': 5432},
        umap_2_kwargs={'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean', 'n_components': 2, 'random_state': 3334}
    )

    # calcul des labels
    np.random.seed(13321)
    embedding_clu.compute_labels()

    # description des clusters
    description_clusters = embedding_clu.describe_clusters(emotes=emotes_channel_name + emotes_global_name).sort_values('published_datetime_min', ascending=True)
    description_clusters.to_csv(os.path.join(save_directory, 'description_clusters.csv'), index=False, encoding='utf-8', decimal=',')

    with open(os.path.join(save_directory, 'print_clusters.txt'), 'w') as f:
        embedding_clu.print_cluster(description_clusters.labels_.to_list(), file=f)

    clustering_columns = ['message_tokenized_length_mean', 'message_tokenized_length_std',
                          'elapsed_time_label_max', 'elapsed_time_label_std',
                          'elapsed_time_label_median', 'count_mention_mean',
                          'endswith_question_mean',
                          'endswith_exclamation_mean',
                          'startswith_command_mean', 'count_emotes_mean',
                          'count_messages_count', 'username_nunique',
                          'silhouette_mean', 'silhouette_std', 'first_word_mean',
                          'term_1_freq', 'term_2_freq',
                          'term_3_freq', 'terms_nunique_mean']

    kmeans_kwargs = {'n_clusters': 20, 'random_state': 129}
    kmeans_model = KMeans(**kmeans_kwargs)
    kmeans_data = description_clusters[clustering_columns]

    kmeans_data_scaled = StandardScaler().fit_transform(kmeans_data)
    kmeans_result = kmeans_model.fit(kmeans_data_scaled)
    description_clusters['km_labels'] = kmeans_result.labels_

    description_clusters_km = description_clusters.groupby('km_labels')[clustering_columns].mean().reset_index()
    description_clusters_km['label_count'] = description_clusters.groupby('km_labels')[clustering_columns].size()
    description_clusters_km.to_csv(os.path.join(save_directory, 'description_clusters_with_km.csv'), index=False, encoding='utf-8', decimal=',')

    pd_corresponsance_hdbscan_km = description_clusters[['labels_', 'km_labels']]
    pd_corresponsance_hdbscan_km = pd_corresponsance_hdbscan_km.rename(columns={'labels_': 'labels'})

    raw_data = pd.merge(embedding_clu.embedding.pd_stats, pd_corresponsance_hdbscan_km, on='labels')
    raw_data.to_csv(os.path.join(save_directory, 'pd_stats.csv'), index=False, encoding='utf-8', decimal=',')

    # export graphs
    p = embedding_clu.plot_coords(with_index=True)
    p.save(os.path.join(save_directory, 'ggplot_plot_coords_with_index.png'))

    p = embedding_clu.plot_x_over_time(x='x')
    p.save(os.path.join(save_directory, 'ggplot_plot_x_over_time.png'))

    p = embedding_clu.plot_labels_over_time()
    p.save(os.path.join(save_directory, 'ggplot_plot_labels_over_time.png'))

    fig = embedding_clu.plotly_labels_x(description_clusters.labels_.tolist())
    fig.write_html(os.path.join(save_directory, 'plotly_labels_x.html'))

    fig = embedding_clu.plotly_mean_labels(description_clusters.labels_.tolist())
    fig.write_html(os.path.join(save_directory, 'plotly_mean_labels.html'))

    fig = embedding_clu.plotly_labels_x_over_time(labels_=description_clusters.labels_.tolist(), include_all_labels=True)
    fig.write_html(os.path.join(save_directory, 'plotly_labels_x_over_time.html'))

    fig = embedding_clu.topic_model.visualize_hierarchy(topics=description_clusters[description_clusters.labels_ != -1].labels_.tolist())
    fig.write_html(os.path.join(save_directory, 'visualize_hierarchy.html'))

    fig = embedding_clu.topic_model.visualize_heatmap(description_clusters[description_clusters.labels_ != -1].labels_.tolist())
    fig.write_html(os.path.join(save_directory, 'visualize_heatmap.html'))


if __name__ == '__main__':
    in_file = '/mnt/twitchat/data/raw/valid_zerator_squeezie_samueletienne_ponce_mistermv_jeanmassiet_domingo_blitzstream_antoinedaniellive_doigby_etoiles.pkl'
    tokenizer_file = '/mnt/twitchat/models/tokenizers/all_streamers.json'
    model_path_or_name = "/mnt/twitchat/models/convbert-small-mlm-simcse-24e/"

    date = '2021-11-04'
    limit = None
    # scales = [None, 300, 600, 1800, 3600]
    scales = [None, 600, 1800, 3600]
    clustering_directory = "/media/data/Projets/twitchat-ds/data/clustering/jeanmassiet_20211104/"

    for scale in scales:
        main(in_file, tokenizer_file, model_path_or_name, date, limit, scale, clustering_directory)
