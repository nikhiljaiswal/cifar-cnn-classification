class Config:
    # Dataset
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.247, 0.243, 0.261)
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 0.015
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5
    
    # Model
    NUM_CLASSES = 10
    DROPOUT_VALUE = 0.05 