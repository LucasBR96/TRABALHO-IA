import pandas as pd
import text_cleaning as tc

if __name__ == '__main__':

    df = pd.read_csv( "DATA/2019-05-28_portuguese_hate_speech_binary_classification.csv" )
    tweets = df.head( n = 50 )[ "text" ]
    freq_dict = {}

    for _ , rec in tweets.items():
        text = tc.normalize_str( rec )
        text = tc.remove_special_words( text )
        for word in text.split():
            freq_dict[ word ] = freq_dict.get( word , 0 ) + 1 

    lst = sorted( freq_dict.keys() , key = lambda x: freq_dict[ x ] )
    for word in lst:
        print( f"{word} {freq_dict[word]}")

