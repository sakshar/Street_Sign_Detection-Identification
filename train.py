from lib import train
from lib import preprocessing
from lib import frontend
from lib import utils
from lib import backend

def _main_(args):
    
    train.training(args)

if __name__ == '__main__':
    import sys
    #print(sys.argv[1])
    args = sys.argv[1:]
    _main_(args)