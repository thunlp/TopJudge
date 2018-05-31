from .model import *

match_list = {
    "CNN": CNN,
    "MultiLSTM": MultiLSTM,
    "CNNSeq": CNNSeq,
    "MultiLSTMSeq": MultiLSTMSeq,
    "Article": Article,
    "LSTM": LSTM,
    "ArtFact": NNFactArt,
    "ArtFactSeq": NNFactArtSeq,
    "Pipeline": Pipeline,
    "HLSTMSeq": HLSTMSeq
}


def get_model(model_name, config, usegpu):
    if model_name in match_list.keys():
        net = match_list[model_name](config, usegpu)
        return net
    else:
        gg
