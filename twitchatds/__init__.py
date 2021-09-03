"""Top-level package for twitchat-ds."""
import os
import io
import re
import glob
import logging
from typing import List
from pathlib import Path
from itertools import chain
import transformers
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.training_args import TrainingArguments
from transformers.utils.dummy_tokenizers_objects import PreTrainedTokenizerFast
from transformers import (MobileBertConfig, MobileBertForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer,
                          PrinterCallback)
from twitchatds.hf_utils import DatasetArguments, FileCallback
import twitch
import tcd
from tcd.settings import Settings

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.implementations import SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing

from datasets import Dataset

__author__ = """Lincoln"""
__email__ = 'francois.vieille@mel.lincoln.fr'
__version__ = '0.1.0'

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

COL_NAMES = ['published_datetime', 'channel', 'username', 'channel_2', 'message']
COL_DTYPES = dict.fromkeys(COL_NAMES, str)
BROADCASTER_ID = {'blitzstream': 49632767, 'jeanmassietaccropolis': 117011503}
_REGEX_MENTION = '@\S+' # noqa
_REGEX_URL = '(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?' # noqa
Settings(str(Path.home()) + '/.config/tcd/settings.json', reference_filepath=f'{os.path.dirname(os.path.abspath(tcd.__file__))}/settings.reference.json')


def global_emotes() -> List:
    helix_api = twitch.Helix(client_id=Settings().config['client_id'], client_secret=Settings().config['client_secret'], use_cache=True)
    return helix_api.api.get('chat/emotes/global').get('data', [])


def channel_emotes(broadcaster_id: int) -> List:
    helix_api = twitch.Helix(client_id=Settings().config['client_id'], client_secret=Settings().config['client_secret'], use_cache=True)
    return helix_api.api.get('chat/emotes', {'broadcaster_id': broadcaster_id}).get('data', [])


def _extract_emote_name(emotes: List) -> List[str]:
    return [infos.get('name') for infos in emotes]


def replace_mention(text: str, replace='<mention>') -> str:
    regex_mention = re.compile(r'(' + _REGEX_MENTION + ')')
    text = re.sub(regex_mention, replace, text)
    return text


def replace_url(text: str, replace='<url>') -> str:
    regex_mention = re.compile(r'%s' % _REGEX_URL)
    text = re.sub(regex_mention, replace, text)
    return text


def read_twitchat_csv(csv_file: str) -> pd.DataFrame:
    regex_backslash = re.compile(r'(\\)[^"]', re.M)
    regex_backslash_end = re.compile(r'(\\)"$', re.M)

    string_io = io.StringIO()

    csv = open(csv_file, 'r')
    lines = csv.read()
    csv.close()

    lines = re.sub(regex_backslash, r'\g<1>\\\\', lines)
    lines = re.sub(regex_backslash_end, r'\\\\"', lines)

    string_io.writelines(lines)
    string_io.seek(0)

    dataframe = pd.read_csv(string_io, sep=";", quoting=2, quotechar='"', doublequote=True, header=None, names=COL_NAMES, parse_dates=[0], escapechar="\\", lineterminator="\n")
    return dataframe


def _group_messages_by_length(x: pd.Series, max_length: int, offset: int = 0) -> List[int]:
    group_index = 0
    sum_ = 0
    group = []
    for val in x:
        if sum_ + val + offset > max_length:
            group_index += 1
            sum_ = 0
        sum_ += val
        group.append(group_index)
    return group


def _tokenizer_messages(messages: pd.Series, tokenizer: Tokenizer) -> pd.Series:
    input_ids_list = [b.ids for b in tokenizer.encode_batch(messages.tolist())]
    return list(chain.from_iterable(input_ids_list))


def prepare_data(csv_path: str, broadcasters: List[str]) -> pd.DataFrame:

    filenames = []
    for broadcaster in broadcasters:
        filenames += glob.glob(f'{csv_path}/{broadcaster}.csv*')

    pd_datas = []
    for filename in filenames:
        try:
            logger.debug(f"Reading {filename} ...")
            pd_datas.append(read_twitchat_csv(filename))
        except Exception as err:
            logger.log(err)

    pd_datas_filter = [d for d in pd_datas if d.published_datetime.dtype == "datetime64[ns]"]
    pd_datas_filter = [d[~d.published_datetime.isnull()] for d in pd_datas_filter]
    pd_datas_filter = [d[~d.channel.isnull()] for d in pd_datas_filter]
    pd_datas_filter = [d[~d.message.isnull()] for d in pd_datas_filter]

    pd_data = pd.concat(pd_datas_filter)
    pd_data = pd_data.sort_values(['channel', 'published_datetime'])
    pd_data['published_date'] = pd_data.published_datetime.dt.floor('d')

    pd_data['count_mention'] = pd_data.message.str.count(f'{_REGEX_MENTION}')
    pd_data['message_length'] = pd_data.message.str.len()
    pd_data['count_url'] = pd_data.message.str.count(f'{_REGEX_URL}')

    return pd_data


def prepare_data_for_tokenization(pd_data: pd.DataFrame, max_length: int = 500, mention_filter: int = 3, count_url_filter: int = 3, time_window_freq: str = '5s') -> pd.DataFrame:

    pd_data = pd_data[pd_data.count_mention <= mention_filter]
    pd_data = pd_data[pd_data.count_url <= count_url_filter]

    pd_data['message_clean'] = pd_data.message.apply(replace_mention).apply(replace_url)
    pd_data['time_window'] = pd_data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq=time_window_freq, origin='start', dropna=True)]).ngroup()
    pd_data['message_length_window'] = pd_data.groupby('time_window')['message_length'].transform(lambda x: _group_messages_by_length(x, max_length))

    pd_data = pd_data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq=time_window_freq, origin='start', dropna=True), pd.Grouper(key='message_length_window')]).agg({
        'message_clean': ' '.join,
        'message_length': 'sum',
        'count_url': 'sum',
        'username': 'count',
    }).reset_index().rename(columns={'username': 'count_messages'})

    return pd_data


def prepare_data_for_mlm(pd_data: pd.DataFrame, tokenizer: Tokenizer, max_length: int = 500, mention_filter: int = 3, count_url_filter: int = 3, time_window_freq: str = '5s') -> pd.DataFrame:

    pd_data = pd_data[pd_data.count_mention <= mention_filter]
    pd_data = pd_data[pd_data.count_url <= count_url_filter]

    pd_data['message_clean'] = pd_data.message.apply(replace_mention).apply(replace_url)
    pd_data['time_window'] = pd_data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq=time_window_freq, origin='start', dropna=True)]).ngroup()
    encoded_batch = tokenizer.encode_batch(pd_data.message_clean)
    pd_data['input_ids'] = [e.ids for e in encoded_batch]
    pd_data['message_tokenized_length'] = [len(e) for e in encoded_batch]
    pd_data['message_tokenized_length_window'] = pd_data.groupby('time_window')['message_tokenized_length'].transform(lambda x: _group_messages_by_length(x, max_length, offset=0))

    pd_data = pd_data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq=time_window_freq, origin='start', dropna=True), pd.Grouper(key='message_tokenized_length_window')]).agg(
        message_clean=('message_clean', ' '.join),
        input_ids=('input_ids', lambda input_ids_list: list(chain.from_iterable(input_ids_list))),
        message_length=('message_length', 'sum'),
        message_tokenized_length=('message_tokenized_length', 'sum'),
        count_url=('count_url', 'sum'),
        count_messages=('username', 'count')
    ).reset_index()

    return pd_data


def build_special_tokens() -> List[str]:
    special_tokens = _extract_emote_name(global_emotes())
    for broadcaster_id in BROADCASTER_ID.values():
        special_tokens += _extract_emote_name(channel_emotes(broadcaster_id))

    special_tokens += ['<mask>', '<sep>', '<unk>', '<cls>', '<pad>', '<mention>', '<url>']

    return special_tokens


def train_tokenizer(pd_data: pd.DataFrame, vocab_size: int, special_tokens: List[str] = []) -> Tokenizer:

    message_clean = pd_data['message_clean'].tolist()
    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.train_from_iterator(
        message_clean,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    tokenizer.post_processor = TemplateProcessing(
        single="<cls> $0 <sep>",
        pair="<cls> $A <sep> $B:1 <sep>:1",
        special_tokens=[
            ("<cls>", tokenizer.token_to_id('<cls>')),
            ("<sep>", tokenizer.token_to_id('<sep>'))
        ],
    )

    return tokenizer


def train_mlm(ds_data: Dataset, tokenizer: PreTrainedTokenizerFast, training_args: TrainingArguments) -> transformers.Trainer:

    mobilebert_config = MobileBertConfig(
        vocab_size=tokenizer.vocab_size,
        sep_token_id=tokenizer.sep_token,
        pad_token_id=tokenizer.pad_token,
        cls_token_id=tokenizer.cls_token,
        hidden_size=128
    )

    mobilebert_model = MobileBertForMaskedLM(config=mobilebert_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=mobilebert_model,
        args=training_args,
        train_dataset=ds_data["train"],
        eval_dataset=ds_data["test"],
        data_collator=data_collator
    )

    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(FileCallback(training_args.output_dir))

    if training_args.do_train:
        try:
            trainer.train(training_args.resume_from_checkpoint)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrup: saving model anyway")

        # saving
        tokenizer.save_pretrained(training_args.output_dir)
        trainer.save_model(output_dir=training_args.output_dir)
        if trainer.is_world_process_zero():
            trainer.save_state()

    return trainer
