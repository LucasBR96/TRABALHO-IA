import pandas as pd
import text_cleaning as tc

if __name__ == '__main__':

    df = pd.read_csv( "DATA/2019-05-28_portuguese_hate_speech_binary_classification.csv" )
    tweets = df[ "text" ]
    freq_words_dict = {}
    freq_tags_dict = {}

    for _ , rec in tweets.items():
        norm = tc.clear_text(rec)
        words, tags = tc.tokenize_text(norm)

        for word in words:
            freq_words_dict[ word ] = freq_words_dict.get( word , 0 ) + 1 
        
        for tag in tags:
            freq_tags_dict[ tag ] = freq_tags_dict.get( tag , 0 ) + 1 

    words_list = sorted( freq_words_dict.keys() , key = lambda x: freq_words_dict[ x ], reverse=True)
    words = " ".join(words_list[:1001])
    
    hashtags_list = sorted( freq_tags_dict.keys() , key = lambda x: freq_tags_dict[ x ], reverse=True)
    hashtags = " ".join(hashtags_list[:101])

    f = open( 'DATA/dummy.txt', 'w')

    f.write(words)
    f.write('\n')
    f.write(hashtags)

    f.close()