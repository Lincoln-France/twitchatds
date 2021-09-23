"""Console script for twitchatds."""
from typing import List
import sys
import os
import argparse
import logging
import logging.handlers
import logging.config
import pandas as pd
from twitchatds import (prepare_data, prepare_data_for_tokenization,
                        prepare_data_for_electra_training,
                        prepare_data_for_mlm, train_tokenizer,
                        build_special_tokens, train_mlm)

logger = logging.getLogger(__name__)


def setup_logging(logfilename: str = None):
    dict_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
            'logfile': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': logfilename
            }
        },
        'loggers': {
            'twitchatds': {
                'handlers': ['default', 'logfile'],
                'level': 0,
                'propagate': True
            },
            'transformers': {
                'handlers': ['logfile'],
                'level': 0,
                'propagate': True
            }
        }
    }

    logging.config.dictConfig(dict_config)

    return


def main():
    """Console script for twitchatds."""
    # todo: argument for logfilename
    setup_logging(logfilename='logs/twitchatds.log')
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(dest="task")
    parser_data = subparser.add_parser('data')
    parser_data.add_argument('-p', '--csv-path', type=str, required=True)
    parser_data.add_argument('-c', '--channel', type=str, nargs='+', required=True)
    parser_data.add_argument('-o', '--out-file', type=str, required=True)
    parser_data.set_defaults(func=prepare_data_task)

    parser_trainer_tokenizer = subparser.add_parser('train_tokenizer')
    parser_trainer_tokenizer.add_argument('-f', '--in-file', type=str, required=True)
    parser_trainer_tokenizer.add_argument('-o', '--out-file', type=str, required=True)
    parser_trainer_tokenizer.add_argument('-s', '--vocab-size', type=int, default=8000)
    parser_trainer_tokenizer.add_argument('-l', '--max-length', type=int, default=500)
    parser_trainer_tokenizer.add_argument('--time-window-freq', type=str, default='5s')
    parser_trainer_tokenizer.add_argument('--mention-filter', type=int, default=3)
    parser_trainer_tokenizer.add_argument('--count-url-filter', type=int, default=3)
    parser_trainer_tokenizer.set_defaults(func=train_tokenizer_task)

    parser_electra_data = subparser.add_parser('electra_data')
    parser_electra_data.add_argument('-f', '--in-file', type=str, required=True)
    parser_electra_data.add_argument('-d', '--out-directory', type=str, required=True)
    parser_electra_data.add_argument('-l', '--max-length', type=int, default=500)
    parser_electra_data.add_argument('--time-window-freq', type=str, default='5s')
    parser_electra_data.add_argument('--mention-filter', type=int, default=3)
    parser_electra_data.add_argument('--count-url-filter', type=int, default=3)
    parser_electra_data.set_defaults(func=export_electra_data)

    parser_tokenize = subparser.add_parser('tokenize')
    parser_tokenize.add_argument('-f', '--file', type=str, required=True)
    parser_tokenize.add_argument('--add_special_tokens', action='store_true')
    parser_tokenize.add_argument('inputs', type=str, nargs='+')
    parser_tokenize.set_defaults(func=tokenize_task)

    parser_train_mlm = subparser.add_parser('train_mlm', add_help=False)
    parser_train_mlm.set_defaults(func=train_mlm_task)

    args, argv = parser.parse_known_args()
    if args.task in ['train_mlm']:
        args.func(argv)
    else:
        dict_args = vars(args).copy()
        dict_args.pop('func')
        dict_args.pop('task')
        args.func(**dict_args)

    return 0


def prepare_data_task(csv_path: str, channel: List[str], out_file: str):
    data = prepare_data(csv_path, broadcasters=channel)
    return data.to_pickle(out_file)


def train_tokenizer_task(in_file: str, out_file: str, vocab_size: int, max_length: int, time_window_freq: str, mention_filter: int, count_url_filter: int):
    pd_data = prepare_data_for_tokenization(
        pd_data=pd.read_pickle(in_file),
        max_length=max_length,
        mention_filter=mention_filter,
        count_url_filter=count_url_filter,
        time_window_freq=time_window_freq
    )
    special_tokens = build_special_tokens()
    tok = train_tokenizer(pd_data, vocab_size, special_tokens)
    return tok.save(out_file)


def tokenize_task(file: str, add_special_tokens: bool, inputs: List[str]):
    from tokenizers import Tokenizer
    tokens_to_print = [s.tokens for s in Tokenizer.from_file(file).encode_batch(inputs, add_special_tokens=add_special_tokens)]
    for input, toks in zip(inputs, tokens_to_print):
        print(f'{input}: {toks}')
    return tokens_to_print


def export_electra_data(in_file: str, out_directory: str, max_length: int, time_window_freq: str, mention_filter: int, count_url_filter: int):
    pd_data = prepare_data_for_electra_training(
        pd_data=pd.read_pickle(in_file),
        max_length=max_length,
        mention_filter=mention_filter,
        count_url_filter=count_url_filter,
        time_window_freq=time_window_freq
    )

    pd_data_channel = pd_data.groupby('channel')

    for name, group in pd_data_channel:
        filename = os.path.join(out_directory, f'raw_{name}.csv')
        open(filename, 'w').close()
        with open(filename, 'a') as f:
            for doc_name, doc in group.groupby('time_window'):
                f.write('\n'.join(doc['message_clean']))
                f.write('\n\n')
    pass


def train_mlm_task(args):
    from transformers import HfArgumentParser, TrainingArguments, PreTrainedTokenizerFast
    import transformers
    from datasets import Dataset
    from tokenizers import Tokenizer
    from twitchatds.hf_utils import DatasetArguments

    transformers.logging.disable_default_handler()
    parser = HfArgumentParser((DatasetArguments, TrainingArguments))
    parser.prog = 'twitchatds train_mlm'
    if "--help" in args or "-h" in args:
        parser.print_help()
        return

    dataset_args, training_args = parser.parse_args_into_dataclasses(args)

    tokenizer = Tokenizer.from_file(dataset_args.tokenizer_file)
    tokenizer.enable_truncation(max_length=dataset_args.max_length)

    pd_data = prepare_data_for_mlm(
        pd_data=pd.read_pickle(dataset_args.data_file),
        tokenizer=tokenizer,
        max_length=dataset_args.max_length,
        mention_filter=dataset_args.mention_filter,
        count_url_filter=dataset_args.count_url_filter,
        time_window_freq=dataset_args.time_window_freq
    )

    # tokenizer.enable_padding(pad_token='<pad>', pad_id=tokenizer.token_to_id('<pad>'))
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        sep_token='<sep>',
        pad_token='<pad>',
        cls_token='<cls>',
        mask_token='<mask>'
    )

    ds_data = Dataset.from_pandas(pd_data[['input_ids']])
    ds_data = ds_data.shuffle(seed=3352)
    ds_data = ds_data.train_test_split(test_size=0.2, seed=9873)

    trainer = train_mlm(ds_data, fast_tokenizer, training_args)

    return trainer


if __name__ == "__main__":
    sys.exit(main())
