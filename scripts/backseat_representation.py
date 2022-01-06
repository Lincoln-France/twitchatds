"""
Figure out which parameters to use with UMAP
"""
import sys
import pandas as pd
import numpy as np
import umap
from twitchatds import get_backseat
from twitchatds.clustering import TwitchatSimCSEEmbedding
from tokenizers import Tokenizer


def main(in_file: str, tokenizer_file: str, date: str, out_file: str, limit=None):

    all_streams = pd.read_pickle(in_file)
    tokenizer = Tokenizer.from_file(tokenizer_file)
    # villani
    backseat_stream = get_backseat(all_streams, date, tokenizer)

    if limit:
        backseat_stream = backseat_stream.head(limit)

    model_path_or_name = "/mnt/twitchat/models/convbert-small-mlm-simcse-24e/"
    embedding_mlm_simcse = TwitchatSimCSEEmbedding(backseat_stream, model_path_or_name)
    embedding_mlm_simcse.compute_embeddings()

    range_scale_time = [0, 300, 600, 1800, 3600]
    range_umap_n_neighbors = [5, 10, 20, 50, 50, 100]
    range_umap_min_dist = [0.05, 0.1, 0.25, 0.5, 0.8, 0.99]
    list_infos_run = []
    vectors = embedding_mlm_simcse.vectors

    for scale_time in range_scale_time:
        for umap_n_neighbors in range_umap_n_neighbors:
            for umap_min_dist in range_umap_min_dist:
                print("Processing scale time {} / n_neighbors {} / min_dist {}".format(scale_time, umap_n_neighbors, umap_min_dist))
                if scale_time > 0:
                    embedding_mlm_simcse.pd_stats['diff_published_datetime'] = (embedding_mlm_simcse.pd_stats.published_datetime - embedding_mlm_simcse.pd_stats.published_datetime.min()).dt.total_seconds()
                    diff_published_time_vector = (embedding_mlm_simcse.pd_stats['diff_published_datetime'] / scale_time).tolist()
                    datetime_vectors = np.transpose([diff_published_time_vector])
                    umap_1_kwargs = {'n_neighbors': umap_n_neighbors, 'min_dist': umap_min_dist, 'metric': 'cosine', 'n_components': 10}
                    umap_2_kwargs = {'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean', 'n_components': 2}
                    representation_model = umap.UMAP(**umap_1_kwargs)
                    vectors_after_umap_1 = representation_model.fit_transform(vectors)
                    vectors_before_umap_2 = np.concatenate((vectors_after_umap_1, datetime_vectors), axis=1)
                    vectors_after_umap_2 = umap.UMAP(**umap_2_kwargs).fit_transform(vectors_before_umap_2)
                    pd_coords = pd.DataFrame(vectors_after_umap_2, columns=['x', 'y'])
                else:
                    umap_1_kwargs = {'n_neighbors': umap_n_neighbors, 'min_dist': umap_min_dist, 'metric': 'cosine', 'n_components': 2}
                    assert umap_1_kwargs['n_components'] == 2
                    representation_model = umap.UMAP(**umap_1_kwargs)
                    vectors_after_umap_1 = representation_model.fit_transform(vectors)
                    pd_coords = pd.DataFrame(vectors_after_umap_1, columns=['x', 'y'])

                pd_info_run = pd.concat([embedding_mlm_simcse.pd_stats['index'], pd_coords], axis=1)
                pd_info_run['scale_time'] = scale_time
                pd_info_run['umap_n_neighbors'] = umap_n_neighbors
                pd_info_run['umap_min_dist'] = umap_min_dist
                list_infos_run.append(pd_info_run)

    pd_infos_run: pd.DataFrame = pd.concat(list_infos_run)

    pd_infos_run['channel'] = 'jeanmassiet'
    pd_infos_run['date'] = date

    pd_infos_run.to_pickle(out_file)


if __name__ == '__main__':

    # default params
    in_file = '/mnt/twitchat/data/raw/valid_zerator_squeezie_samueletienne_ponce_mistermv_jeanmassiet_domingo_blitzstream_antoinedaniellive_doigby_etoiles.pkl'
    tokenizer_file = '/mnt/twitchat/models/tokenizers/all_streamers.json'
    date = '2021-11-04'
    out_file = '/media/data/Projets/twitchat-ds/data/interim/umap_jeanmassiet_20211104.pkl'
    limit = None

    main(in_file, tokenizer_file, date, out_file, limit)
    sys.exit(0)
