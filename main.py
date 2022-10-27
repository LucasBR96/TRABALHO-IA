import data_sampling as ds
import model as md

set_train = ds.HateSpeechDataset()
set_eval  = ds.HateSpeechDataset( True )
md.learning_loop( set_train , set_eval )