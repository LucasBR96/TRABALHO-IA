import re
import unicodedata as und

EXCESS_WHITE = r"( {2,}|\t+|\n+)"
PONCTUATION  = r"[\?!,\.\":]"
LINK         = r"(https?:?//\S+|@\w+)"

HASHTAG      = r"(?<!\S)#\S+"

def normalize_str( entry_str : str ) -> str:

    text = entry_str.lower()

    #----------------------------------------------------------------
    # removing diacritics
    text = und.normalize( 'NFD' , text )
    shaved = ''.join( c for c in text if not und.combining( c ) )
    text   = und.normalize( 'NFC' , shaved )

    #-----------------------------------------------------------------
    #removing unecessary characters
    text = re.sub( EXCESS_WHITE , " " , text )  # excess whitespace
    text = re.sub( PONCTUATION , "" , text )    # ponctuation 
    
    return text

def remove_special_words( entry_str : str ) -> str:
    return re.sub( LINK , "" , entry_str )

norm = normalize_str( "A afronta aos portugueses não pára! https://t.co/mGEV4CjFRP#RenovarPortugal #PNR #Portugal #Imigração https://t.co/26j98jdIYH" )
print( norm )

print( remove_special_words( norm ) )