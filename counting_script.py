import pandas as pd
import text_cleaning as tc
import sys

NUM_WORDS = 1000
NUM_TAGS = 100
ONLY_POS = False

if __name__ == '__main__':

    df = pd.read_csv( "DATA/hate_speech_pt.csv" )
    tweets = df[ "text" ]
    if ONLY_POS:
        tweets = df[ ["text","hatespeech_comb"] ]

    freq_words_dict = {}
    freq_tags_dict = {}

    for rec in tweets.to_numpy():

        txt = rec
        should_count = True
        if ONLY_POS:
            txt = rec[ 0 ]
            should_count = bool( rec[ 1 ] )
        
        if not should_count:
            continue

        norm = tc.clear_text(txt)
        words, tags = tc.tokenize_text(norm)

        for word in words:
            freq_words_dict[ word ] = freq_words_dict.get( word , 0 ) + 1 
        
        for tag in tags:
            freq_tags_dict[ tag ] = freq_tags_dict.get( tag , 0 ) + 1 

    words_list = sorted( freq_words_dict.keys() , key = lambda x: freq_words_dict[ x ], reverse=True)
    words = " ".join(words_list[:NUM_WORDS])
    
    hashtags_list = sorted( freq_tags_dict.keys() , key = lambda x: freq_tags_dict[ x ], reverse=True)
    hashtags = " ".join(hashtags_list[:NUM_TAGS])

    f = open( 'DATA/dummy.txt', 'w')
    f.write(f"{NUM_WORDS}\n")
    f.write(words)
    f.write('\n')
    f.write(f"{NUM_TAGS}\n")
    f.write(hashtags)
    f.close()