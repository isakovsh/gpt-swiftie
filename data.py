import re
import lyricsgenius
import torch 
from torch.utils.data import  Dataset 



def download_data():
    """ Downloads data """
    genius = lyricsgenius.Genius("YOUR_API_KEY") 
    artist = genius.search_artist("Taylor Swift", max_songs=50, sort="popularity")
    try:

        with open("taylor_swift.txt", "w", encoding="utf-8") as f:
            for song in artist.songs:
                f.write(f"{song.title}\n{song.lyrics}\n\n")
    except Exception as e:
        return e

    return "Sucsesfully downloaded data"



    