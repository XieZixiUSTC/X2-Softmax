config = dict(
    NET="r50",
    FP16=True,
    EMBEDDING_SIZE=512,

    LOSS=None,
    LOSSPARAMETERS=None,
    OUTPUT=None,

    SEED=2048,

    BATCH_SIZE=128,
    NUM_WORKERS=2,

    NUM_CLASS=None,
    NUM_IMG=None,
    STEPSPEREPOCH=None,

    TRAIN_ROOT="D:\\Dataset\\MS1MV3\\ms1mv3",

    NUM_EPOCH=26,

    OPTIMIZER="SGD",
    LR=0.1,
    MOMENTUM=0.9,
    WEIGHT_DECAY=5e-4,

    LR_SCHEDULER="MULTISTEP",
    MILESTONES=[8, 14, 20, 25],
    GAMMA=0.1,
)

