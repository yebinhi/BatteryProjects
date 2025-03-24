import torch


class Config:
    INPUT_SIZE = 5
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    D_MODEL = 32
    OUTPUT_DIM = 1
    INPUT_DIM = 5
    LR = 0.001
    EPOCHS = 200
    BATCH_SIZE = 32
    STEP_SIZE = 10
    GAMMA = 0.99
    TRAIN_SPLIT = 0.8
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    MODEL_SAVE_PATH = '../../models/'
    FIGURE_SAVE_PATH = '../../figs/'
    FILE_PATHS = [
        "../../data/soh/Normalized_Dataset3.csv",
        "../../data/soh/Normalized_Dataset5.csv",
        "../../data/soh/Normalized_oxford.csv",
    ]