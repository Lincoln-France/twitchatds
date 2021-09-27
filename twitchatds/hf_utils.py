import os
import json
import datetime
from dataclasses import dataclass, field
from transformers import TrainerCallback


@dataclass
class DatasetArguments:
    data_file: str = field(
        metadata={"help": "Pandas dataframe pickle"}
    )

    tokenizer_file: str = field(
        metadata={"help": "Json file of tokenizer"}
    )

    mention_filter: int = field(
        default=3
    )

    count_url_filter: int = field(
        default=3
    )

    time_window_freq: str = field(
        default='5s'
    )

    max_length: int = field(
        default=250,
        metadata={"help": "max length of grouping message"}
    )


@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


class FileCallback(TrainerCallback):
    """ Quick callback to write in file """
    def __init__(self, log_path: str):
        self.train_log_file = os.path.join(log_path, 'train_metrics.log')
        self.eval_log_file = os.path.join(log_path, 'eval_metrics.log')
        for log_file in [self.train_log_file, self.eval_log_file]:
            with open(log_file, 'w') as f:
                f.write("[")

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if list(logs.keys())[0].startswith('eval'):
            log_file = self.eval_log_file
        else:
            log_file = self.train_log_file
        with open(log_file, 'a') as f:
            logs.update({"time": datetime.datetime.now().astimezone().isoformat()})
            f.write(json.dumps(logs) + ",\n")

    def on_train_end(self, args, state, control, **kwargs):
        for log_file in [self.train_log_file, self.eval_log_file]:
            with open(log_file, 'a') as f:
                f.write("]")
