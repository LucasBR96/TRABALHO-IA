import re

EXCESS_WHITE = r"( {2,}|\t+|\n+)"
PONCTUATION  = r"[\?!,\.]"

def normalize_str( entry_str : str ):

    text = entry_str.lower()

    #-----------------------------------------------------------------
    #removing unecessary characters
    text = re.sub( EXCESS_WHITE , " " , text )  # excess whitespace
    text = re.sub( PONCTUATION , "" , text ) 
    
    print( text )

normalize_str( "@_Ioen imagino...  ow mandei fazer um body opressor pra Duda! hahahaha" )