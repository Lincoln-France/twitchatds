"""
Script to create visualisation of topics over time
"""
# import
import os
import pandas as pd
import numpy as np
import datetime as dt
from tokenizers import Tokenizer
import bar_chart_race as bcr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import rgb2hex


def get_window_stats(start_window, end_window, pd_messages, description_cluster, exclude=[]):
    window_data = pd_messages[(pd_messages.published_datetime >= start_window) & (pd_messages.published_datetime < end_window)]
    windows_stats = window_data.groupby(['regroupement_id', 'definition', 'km_labels', 'labels', 'color']).agg({
        'message': [np.random.choice, 'count']
    }).reset_index()
    windows_stats.columns = ["_".join([el for el in a if el != '']) for a in windows_stats.columns.to_flat_index()]
    windows_stats = windows_stats.merge(description_cluster[['labels', 'best_textual_description']], on='labels')
    windows_stats = windows_stats[~windows_stats.definition.isin(exclude)]
    windows_stats = windows_stats[windows_stats.best_textual_description != '']
    return windows_stats.sort_values('message_count', ascending=False)


def main(tokenizer_file, scale, debut_stream_str, fin_stream_str, clustering_directory, window_time_sec, top_n_ploted, exclude, chart_race):
    if scale is None:
        scale_directory = "no_scale"
    else:
        scale_directory = "scale_{}".format(scale)
    save_directory = os.path.join(clustering_directory, scale_directory)
    raw_data = pd.read_csv(os.path.join(save_directory, 'pd_stats.csv'), decimal=',', converters={"input_ids": lambda x: [int(el) for el in x.strip("[]").replace("'", "").split(", ")]})
    description_cluster = pd.read_csv(os.path.join(save_directory, 'description_clusters.csv'), decimal=',')
    tokenizer = Tokenizer.from_file(tokenizer_file)
    correspondance_km_hdbscan = raw_data[['labels', 'km_labels']]
    correspondance_km_hdbscan = correspondance_km_hdbscan[~correspondance_km_hdbscan.duplicated()]
    cmap = cm.get_cmap('Paired')

    # pour 300
    if scale == 300:
        regroupements = {
            '2': {'km_labels': [8], 'definition': 'LUL', 'color': rgb2hex(cmap(0))},
            '3': {'km_labels': [9, 15], 'definition': 'Quizz', 'color': rgb2hex(cmap(1))},
            '4': {'km_labels': [10], 'definition': 'Commandes', 'color': rgb2hex(cmap(2))},
            '5': {'km_labels': [13, 4], 'definition': 'Emotes', 'color': rgb2hex(cmap(3))},
            '6': {'km_labels': [11, 17], 'definition': 'Exclamation', 'color': rgb2hex(cmap(4))},
            '7': {'km_labels': [6, 3, 16], 'definition': 'Reactions', 'color': rgb2hex(cmap(5))},
            '8': {'km_labels': [2], 'definition': 'Question', 'color': rgb2hex(cmap(6))},
            '9': {'km_labels': [0, 19, 7, 5, 14, 12, 18], 'definition': 'Opinion', 'color': rgb2hex(cmap(7))}
        }
    # pour 600
    elif scale == 600:
        regroupements = {
            '2': {'km_labels': [5], 'definition': 'LUL', 'color': rgb2hex(cmap(0))},
            '3': {'km_labels': [6, 10], 'definition': 'Quizz', 'color': rgb2hex(cmap(1))},
            '4': {'km_labels': [8, 19], 'definition': 'Commandes', 'color': rgb2hex(cmap(2))},
            '5': {'km_labels': [9, 17], 'definition': 'Emotes', 'color': rgb2hex(cmap(3))},
            '6': {'km_labels': [15, 16], 'definition': 'Exclamation', 'color': rgb2hex(cmap(4))},
            '7': {'km_labels': [4, 12, 11, 13], 'definition': 'Reactions', 'color': rgb2hex(cmap(5))},
            '8': {'km_labels': [2], 'definition': 'Question', 'color': rgb2hex(cmap(6))},
            '9': {'km_labels': [18, 7, 14, 1, 0], 'definition': 'Opinion', 'color': rgb2hex(cmap(7))}
        }
    else:
        raise ValueError("Scale not implemented")

    rows = []
    for regroupement_id, values in regroupements.items():
        for km_label in values['km_labels']:
            rows.append({
                'regroupement_id': regroupement_id,
                'km_labels': km_label,
                'definition': values['definition'],
                'color': values['color']
            })
    pd_regroupements = pd.DataFrame(rows)

    debut_stream = dt.datetime.strptime(debut_stream_str, '%Y-%m-%d %H:%M')
    fin_stream = dt.datetime.strptime(fin_stream_str, '%Y-%m-%d %H:%M')

    outfile_name = "{}_{}".format(debut_stream.strftime('%H%M'), fin_stream.strftime('%H%M'))

    raw_data['published_datetime'] = pd.to_datetime(raw_data['published_datetime'])
    pd_messages = pd.merge(raw_data, pd_regroupements, on='km_labels')

    best_textual_description = []
    for i, row in description_cluster.iterrows():
        messages = raw_data[raw_data.labels == row['labels']][['x', 'y', 'input_ids', 'message_tokenized_length', 'count_mention']]
        messages['length_normalized'] = (messages.message_tokenized_length / messages.message_tokenized_length.max())
        messages['distance'] = ((messages.x - messages.x.median())**2 + (messages.y - messages.y.median())**2) * messages['length_normalized'] / (1 + np.exp(-messages['length_normalized']))

        input_ids = messages.sort_values(['count_mention', 'distance'], ascending=True).input_ids.iloc[0][1:-1]
        if len(input_ids) > 20:
            rpz_message = tokenizer.decode(input_ids[:21], skip_special_tokens=False) + '...'
        else:
            rpz_message = tokenizer.decode(input_ids, skip_special_tokens=False)

        if len(rpz_message) > 20:
            tmp = rpz_message.split(' ')
            ix_spit = len(tmp) // 2 + 1
            rpz_message = ' '.join(tmp[0:ix_spit]) + '\n' + ' '.join(tmp[ix_spit:])

        legend = ''

        if row['first_word_mean'] > 0.5:
            legend = row['first_word']
        else:
            to_concat = []
            term_x = ['term_1', 'term_2', 'term_3']
            term_x_freq = [x + '_freq' for x in term_x]
            for term, term_freq in zip(row[term_x].tolist(), row[term_x_freq].tolist()):
                if isinstance(term, str) and term_freq > 1:
                    to_concat.append(term)
            legend = ' / '.join(to_concat)
        if legend.lower() != rpz_message.lower():
            legend = rpz_message + ' (' + legend + ')'
        best_textual_description.append(legend)

    description_cluster['best_textual_description'] = best_textual_description

    if chart_race:
        datetimes_range = pd.date_range(start=debut_stream, end=fin_stream, freq='300s')
        dfs = []
        for datetime_v in datetimes_range:
            end_window = datetime_v + dt.timedelta(seconds=window_time_sec)
            windows_stats = get_window_stats(datetime_v, end_window, pd_messages, description_cluster, exclude=exclude).head(top_n_ploted)
            windows_stats['start_of_period'] = datetime_v
            dfs.append(windows_stats)

        pd_for_bcr = pd.concat(dfs)
        # pd_for_bcr['longueur'] = pd_for_bcr.best_textual_description.str.len()
        # pd_for_bcr.sort_values('longueur', ascending=False)
        # pd_for_bcr.sort_values('longueur', ascending=False).best_textual_description.tolist()

        df_wide = pd_for_bcr.pivot_table(
            index='start_of_period',
            columns='best_textual_description',
            values='message_count',
            aggfunc='sum'
        ).fillna(0)
        df_values, df_ranks = bcr.prepare_wide_data(df_wide, steps_per_period=1)
        df_values = df_values.fillna(0)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=144, tight_layout=True)
        bcr.bar_chart_race(df_values, filename=os.path.join(save_directory, f'bar_chart_race_{outfile_name}.mp4'),
                           period_length=10000, steps_per_period=50,
                           fixed_max=False, fixed_order=False, n_bars=5,
                           period_fmt='%H:%M',
                           title='Emission BACKSEAT 04/11/2011',
                           filter_column_colors=True,
                           bar_label_size=8,
                           tick_label_size=7,
                           bar_size=.5,
                           shared_fontdict={'family': 'Helvetica', 'weight': 'light'},
                           fig=fig)
    else:
        stream_windows_stats = get_window_stats(
            start_window=debut_stream,
            end_window=fin_stream,
            pd_messages=pd_messages,
            description_cluster=description_cluster,
            # exclude=['LUL', 'Quizz', 'Commandes', 'Emotes', 'Exclamation', 'Reactions', 'Question', 'Opinion']
            exclude=exclude
        ).head(top_n_ploted)

        patches = [mpatches.Patch(color=v['color'], label=v['definition']) for v in regroupements.values()]
        fig = plt.figure(figsize=(10, 10), dpi=144, tight_layout=True)
        plt.barh(stream_windows_stats.best_textual_description, stream_windows_stats.message_count,
                 height=0.5, color=stream_windows_stats.color)
        plt.legend(handles=patches, loc=1)

        fig.savefig(os.path.join(save_directory, f'scale_{scale}_{outfile_name}.png'))


if __name__ == '__main__':
    tokenizer_file = '/mnt/twitchat/models/tokenizers/all_streamers.json'
    limit = None
    clustering_directory = "/media/data/Projets/twitchat-ds/data/clustering/jeanmassiet_20211104/"
    scale = 600
    window_time_sec = scale
    top_n_ploted = 10
    debut_stream_str = '2021-11-04 19:10'
    fin_stream_str = '2021-11-04 22:00'
    chart_race = True
    # exclude = ['LUL', 'Quizz', 'Commandes', 'Emotes']
    exclude = ['LUL', 'Commandes', 'Emotes']

    main(tokenizer_file, scale, debut_stream_str, fin_stream_str, clustering_directory, window_time_sec, top_n_ploted, exclude, chart_race)
