# flake8: noqa
# %%
import os
from pandas.core.algorithms import mode
from plotnine.geoms.geom_point import geom_point
from tokenizers import Tokenizer
import tokenizers
import twitchatds
import glob
from plotnine import ggplot, geom_col, aes, theme, coord_flip, geom_boxplot, geom_point
from plotnine.themes import element_text
import pandas as pd
import sentencepiece as spm
import csv

# %% [markdown]
# # Analyse du chat de la chaine de JeanMassietAccropolis

# %% [markdown]
# ### Identifiant de la chaine

# %% 
twitchatds.BROADCASTER_ID['jeanmassietaccropolis']

# %% [markdown]
# ### Récupération des Emotes

# %%
emotes_channel_name = [infos.get('name') for infos in twitchatds.channel_emotes(twitchatds.BROADCASTER_ID['jeanmassietaccropolis'])]
emotes_channel_name

# %%
emotes_global_name = [infos.get('name') for infos in twitchatds.global_emotes()]
emotes_global_name

# %% [markdown]
# ### Lister les fichiers utiles
csv_path = '/media/data/twitchat-data'
filenames = glob.glob(f'{csv_path}/jeanmassietaccropolis.csv*')
# filenames = glob.glob(f'{csv_path}/*.csv*')

# %% [markdown]
# ### Lecture des données

# %%
datas = []
for f in filenames:
    try:
        datas.append(twitchatds.read_twitchat_csv(f))
    except Exception as err:
        print(f)
        print(err)

# %%
# filtrer les données 
datas_filter = [d for d in datas if d.published_datetime.dtype == "datetime64[ns]"]
datas_filter = [d[~d.published_datetime.isnull()] for d in datas_filter]
datas_filter = [d[~d.channel.isnull()] for d in datas_filter]
datas_filter = [d[~d.message.isnull()] for d in datas_filter]

# %%
data = pd.concat(datas_filter)
data = data.sort_values(['channel', 'published_datetime'])

# %% [markdown] 
# ### quelques statistiques

# %%
data['published_date'] = data.published_datetime.dt.floor('d')

# %%
data_freq_by_day = data.groupby(by='published_date').size().reset_index(name='count').sort_values('published_date')

# %%
ggplot(data_freq_by_day, aes('published_date', 'count')) + geom_col()

# %%
data_freq_by_channel = data.groupby('channel').size().reset_index(name='count').sort_values('channel')

# %%
(ggplot(data_freq_by_channel, aes('channel', 'count'))
+ geom_col() + coord_flip() + theme(axis_text_x = element_text(rotation=45)))

# %% [markdwon]
# ### calcul de stats

# %%
# nombre de mention
data['count_mention'] = data.message.str.count('@\S+')

# %%
# longueur message
data['message_length'] = data.message.str.len()

# %%
# nombre url
data['count_url'] = data.message.str.count(r'(http|ftp|https):\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?')

# %%
data_freq_by_mention = data.groupby('count_mention').size().reset_index(name='count').sort_values('count_mention')
data_freq_by_mention.head(10)

# %% 
data.groupby('count_url').size().reset_index(name='count').sort_values('count')

# %%
data = data[data.count_mention <= 3]
data = data[data.count_url <= 3]

# %%
(data.message_length.mean(), data.message_length.median())

# %%
data_freq_by_len = data.groupby('message_length').size().reset_index(name='count')

# %%
(ggplot(data_freq_by_len, aes('message_length', 'count')) + geom_col())


# %%
data['message_clean'] = data.message.apply(twitchatds.replace_mention).apply(twitchatds.replace_url)

# %%
data['time_window'] = data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq='5s', origin='start', dropna=True)]).ngroup()

# %% 
data = data.sort_values('published_datetime')

# %%
def group_messages(x):
    group_index = 0
    sum_ = 0
    group = []
    for val in x:
        if sum_ + val > 500:
            group_index += 1
            sum_ = 0
        sum_ += val
        group.append(group_index)
    return group

# %%
data['message_length_window'] = data.groupby('time_window')['message_length'].transform(group_messages)


# %%
data_concat = data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq='5s', origin='start', dropna=True), pd.Grouper(key='message_length_window')]).agg({
    'message_clean': ' '.join,
    'message_length': 'sum',
    'count_url': 'sum',
    'username': 'count',
}).reset_index().rename(columns={'username': 'count_messages'})

# %%
(ggplot(data_concat.sample(frac=0.01), aes(x='count_messages', y='message_length')) + geom_point())

# %%
data_concat_freq_by_len = data_concat.groupby('message_length').size().reset_index(name='count')

# %%
(data_concat.message_length.median(), data_concat.message_length.max())

# %%
data_concat[data_concat.message_length == 100]['message_clean'].iloc[10]

# %%
(ggplot(data_concat_freq_by_len, aes('message_length', 'count')) + geom_col())

# %%
data_concat_freq_by_len.sort_values('message_length', ascending=False)


# %% [markdown]
# ### SentencePiece

# %%
data[['message_clean']].to_csv('../data/processed/jeanmassietaccropolis_sentencepiece.txt', index=None, header=None, doublequote=False, escapechar="\\", quoting=csv.QUOTE_NONE)

# %%
user_defined_symbols = emotes_global_name + emotes_channel_name + ['<mention>', '<url>', '<mask>']
user_defined_symbols = list(set(user_defined_symbols))
control_symbols = ['<cls>', '<sep>']


# %%
spm.SentencePieceTrainer.train(input='../data/processed/jeanmassietaccropolis_sentencepiece.txt', model_prefix='../models/vocab/jeanmassietaccropolis_sentencepiece', vocab_size=8000,  user_defined_symbols=user_defined_symbols, control_symbols=control_symbols, pad_piece='<pad>', pad_id=3)


# %%
sp = spm.SentencePieceProcessor(model_file='../models/vocab/jeanmassietaccropolis_sentencepiece.model')

# %%
sp.encode('Coucou Jean éé <mask> ! <s>', out_type=str)

# %%
sp.piece_to_id(['<cls>', '<mask>', '<sep>', '<s>', '</s>', '<mention>', '<url>', '<unk>', '<pad>'])
# sp.decode_ids([0])


# %% [markdown] 
# ### Tokenizer avec HF

# %%
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')

# %%
tokenizer.tokenize('ok[SEP]')

# %%
tokenizer_sp = AutoTokenizer.from_pretrained('xlnet-base-cased')
# %%
tokenizer_sp.tokenize('<s>Bonjour comment <mask> tu vas <s> lkdlf</s>')

# %%
tokenizer_sp.convert_tokens_to_string(tokenizer_sp.tokenize('<s>Bonjour comment <mask> tu vas <s> lkdlf</s>'))

# %%
from twitchatds.tokenization_mobilebert import MobileBertSentencePieceTokenizer

# %%
tokenizer_xlnet = MobileBertSentencePieceTokenizer(
    vocab_file='../models/vocab/jeanmassietaccropolis_sentencepiece.model',
    keep_accents=True,
    additional_special_tokens=['<mask>']
)

# %%
tokenizer_xlnet.tokenize('Salut Accropolis <cls> <mask> ! mdr <s>')
# %%
tokenizer_xlnet.convert_tokens_to_string(tokenizer_xlnet.tokenize("Salut Accropolis éé '' ! "))


# %% [markdown] 
# ### Tokenizer avec HF
from transformers import MobileBertModel, MobileBertForMaskedLM, pipeline, FillMaskPipeline, MobileBertConfig, MobileBertTokenizer

# %%
mobilebert_config = MobileBertConfig(
    vocab_size=tokenizer_xlnet.vocab_size,
    sep_token_id=tokenizer_xlnet.sep_token_id,
    bos_token_id=tokenizer_xlnet.bos_token_id,
    eos_token_id=tokenizer_xlnet.eos_token_id,
    pad_token_id=tokenizer_xlnet.pad_token_id,
    hidden_size=128
)

# %%
mobilebert_model = MobileBertForMaskedLM(config=mobilebert_config)
# mobilebert_model = MobileBertForMaskedLM(config=MobileBertConfig())

# %%
fill_mask = pipeline(
    "fill-mask",
    model=mobilebert_model,
    tokenizer=tokenizer_xlnet
)

# %%
fill_mask("bonjour <mask>")

# %%
tmp_input = tokenizer_xlnet(["test", "lkfg kdf d"], return_tensors='pt', return_token_type_ids=False, padding=True)

# %%
tmp_input

# %%
mobilebert_model(**tmp_input)

# %% [markdown]
# ### Format données entrainement

# %%
from datasets import load_dataset, Dataset
from itertools import chain
from transformers import Trainer, TrainingArguments


# %%
def group_messages(messages, tokenizer):
    input_ids_list = tokenizer(messages.tolist(), padding=False, return_attention_mask=False, return_token_type_ids=False, max_length=250, truncation=True)['input_ids']
    return list(chain.from_iterable(input_ids_list))
    

# %%
data_concat = data.groupby([pd.Grouper(key='channel'), pd.Grouper(key='published_datetime', freq='5s', origin='start', dropna=True), pd.Grouper(key='message_length_window')]).agg(
    message_clean=('message_clean', ' '.join),
    input_ids=('message_clean', lambda msgs: group_messages(msgs, tokenizer=tokenizer_xlnet)),
    message_length=('message_length', 'sum'),
    count_url=('count_url', 'sum'),
    count_messages=('username', 'count')
).reset_index()

# %%
ds = Dataset.from_pandas(data_concat[['input_ids']])
ds = ds.shuffle(seed=3352)
# ds = ds.rename_column("message_clean", "text")
ds = ds.train_test_split(test_size=0.2)

# %%
ds

# %%
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_xlnet, mlm_probability=0.15)

# %%
training_args = TrainingArguments(
    "../models/mobilebert/mlm",
    # evaluation_strategy="step",
    num_train_epochs=1,
    max_steps=60
)

# %%
trainer = Trainer(
    model=mobilebert_model,
    args=training_args,
    train_dataset=ds["train"].train_test_split(train_size=0.1)['train'],
    eval_dataset=ds["test"],
    data_collator=data_collator
)

# %%
trainer.train()
# %%
trainer.evaluate()
# %%
trainer.save_model()

# %%
trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
# %%
tokenizer_xlnet.save_pretrained('../models/mobilebert/mlm')

# %% [markdown]
# ### Usage

# %%
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForPreTraining, AutoModel

# %%
tokenizer_xlnet = MobileBertSentencePieceTokenizer.from_pretrained('../models/mobilebert/mlm')

# %%
mobilebert_model = AutoModelForMaskedLM.from_pretrained('../models/mobilebert/mlm')

# %%
mobilebert_model

# %%
fill_mask = pipeline(
    "fill-mask",
    model=mobilebert_model,
    tokenizer=tokenizer_xlnet
)

# %%
fill_mask('Accro<mask>')

# %%
mobilebert_model = AutoModel.from_pretrained('../models/mobilebert/mlm')

# %% [markdown]
# ### test tokenizer

# %%
from tokenizers.implementations import SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import UnigramTrainer

# %%
tokenizer_tmp = SentencePieceUnigramTokenizer.from_spm('../models/vocab/jeanmassietaccropolis_sentencepiece.model')

# %%
tokenizer_tmp.post_processor = TemplateProcessing(
    single="<cls> $0 <sep>",
    pair="<cls> $A <sep> $B:1 <sep>:1",
    special_tokens=[("<cls>", 4), ("<sep>", 5)],
)

# %%
tokenizer_tmp.encode("<s>test</s><sep><cls>").tokens

# %%
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

# %%
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_tmp)

# %%
fast_tokenizer.tokenize("<sep> ok </s> accropolis <url>", add_special_tokens=True)
# %%
fast_tokenizer("<sep> <cls>ok </s> accropolis <url> ldklskdjf", add_special_tokens=True)


# %%
fast_tokenizer.convert_ids_to_tokens([1583])


# %%
tokenizer = SentencePieceUnigramTokenizer()
data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]
tokenizer.train_from_iterator(data, special_tokens=['[SEP]'])

# %% 
tokenizer.add_special_tokens(["[S]"])

# %%
tokenizer.token_to_id('[S]')

# %%
tokenizer.encode("Simple is better [SEP] than complex [S].").tokens
# %%
tokenizer.train_from_iterator(data, special_tokens=[])
tokenizer.encode("Simple is better [SEP] than complex.").tokens

# %%
fill_mask = pipeline(
    "fill-mask",
    model=mobilebert_model,
    tokenizer=fast_tokenizer
)


# %%
fill_mask('Accro<mask>')

# %% [markdown]
# ## Test des fonctions

# %%
from twitchatds import BROADCASTER_ID, prepare_data, prepare_data_for_tokenization, train_tokenizer, build_special_tokens
from tokenizers import Tokenizer

# %%
pd_data = prepare_data('/media/data/twitchat-data', ['jeanmassietaccropolis'])

# %%
pd_data_for_tok = prepare_data_for_tokenization(pd_data)

# %%
special_tokens = build_special_tokens()

# %%
tokenizer = train_tokenizer(pd_data_for_tok, 8000, special_tokens=special_tokens)

# %%
tokenizer.encode('Coucou les gens, comment va Jean ? ! <mention>').tokens

# %%
tokenizer.save("../models/tokenizers/jeanmassietaccropolis_unigram.json")
# %%
tokenizer = Tokenizer.from_file("../models/tokenizers/jeanmassietaccropolis_unigram.json")

# %%
tmp = tokenizer.encode_batch(["Coucou ceci est une premiere", "Coucou ceci est une deuxième dfdf"])

# %%
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    sep_token='<sep>',
    pad_token='<pad>',
    cls_token='<cls>',
    mask_token='<mask>',
    unk_token='<unk>'
)

# %%
fast_tokenizer.save_pretrained('../models/mobilebert/mlm')
# %%
fast_tokenizer = PreTrainedTokenizerFast.from_pretrained('../models/mobilebert/mlm')

# %%
fast_tokenizer('Coucou Jean !')

# %%
from transformers import MobileBertForMaskedLM

# %%
mobilebert_model = MobileBertForMaskedLM.from_pretrained('../models/mobilebert/mlm')

# %%
fill_mask = pipeline(
    "fill-mask",
    model=mobilebert_model,
    tokenizer=fast_tokenizer
)


# %%
fill_mask("Comment va <mask>")

# %%
from twitchatds import prepare_data_for_mlm
from tokenizers import Tokenizer
import pandas as pd

# %%
tokenizer = Tokenizer.from_file('../models/tokenizers/jeanmassietaccropolis.json')

tokenizer.enable_padding(pad_token='<pad>', pad_id=tokenizer.token_to_id('<pad>'))
# tokenizer.enable_truncation(max_length=)

# %%
pd_data_tmp = pd.read_pickle('../data/raw/jeanmassietaccropolis.pkl')

# %%
pd_data_tmp.shape

# %%
pd_data = prepare_data_for_mlm(
    pd_data=pd_data_tmp,
    tokenizer=tokenizer,
    time_window_freq='10s'
)

# %%
pd_data.message_length.describe()

# %%
pd_data.message_length.quantile(0.9)

# %%
pd_data.count_messages.describe()

# %%
pd_data[pd_data.count_messages > 100]

# %%
