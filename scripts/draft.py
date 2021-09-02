import os
import twitch
import tcd
from tcd.settings import Settings
from pathlib import Path
from twitch import helix

Settings(str(Path.home()) + '/.config/tcd/settings.json', reference_filepath=f'{os.path.dirname(os.path.abspath(tcd.__file__))}/settings.reference.json')

helix_api = twitch.Helix(client_id=Settings().config['client_id'], client_secret=Settings().config['client_secret'], use_cache=True)

blitzstream_broadcaster_id = 49632767

for emote in helix_api.api.get('chat/emotes', {'broadcaster_id': blitzstream_broadcaster_id}).get('data'):
    print(emote.get('name'))

