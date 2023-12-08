import os.path

import bcolz
import numpy as np
import torch
import torchvision.transforms as transforms

from backbones import get_model
from configuration import config

import argparse


def Calculate_features(batch_imgs):
    batch_features = BACKBONE(batch_imgs.to(DEVICE)).cpu()
    return batch_features


def Calculate_Accuracy(features, isSame):
    m = features.shape[0] // 2
    features1 = torch.tensor(features[0::2]).to(DEVICE)
    features2 = torch.tensor(features[1::2]).to(DEVICE)
    cosine = torch.sum(features1 * features2, dim=1) / (torch.norm(features1, dim=1) * torch.norm(features2, dim=1))

    best_acc = 0
    best_th = 1
    labels = torch.tensor(isSame).to(DEVICE)
    for i in range(m):
        th = cosine[i]
        acc = ((cosine > th) == labels).int().sum() / m
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_th, best_acc


def Evaluate(testset):
    idx = 0
    carray = bcolz.carray(rootdir=os.path.join(DATA_ROOT, testset), mode='r')
    isSame = np.load(DATA_ROOT + f'\\{testset}_list.npy')
    features = np.zeros([len(carray), EMBEDDING_SIZE])
    with torch.no_grad():
        while idx + BATCH_SIZE <= len(carray):
            batch = torch.tensor(carray[idx:idx + BATCH_SIZE][:, [2, 1, 0], :, :])
            features[idx:idx + BATCH_SIZE] = Calculate_features(batch)

            idx += BATCH_SIZE
        if idx < len(carray):
            batch = torch.tensor(carray[idx:][:, [2, 1, 0], :, :])
            features[idx:idx + BATCH_SIZE] = Calculate_features(batch)
    th, acc = Calculate_Accuracy(features, isSame)
    print(f"{testset}: acc = {acc:.4f}, the = {th:.4f}")

    f = open(MODEL_ROOT + "\\" + TXT_NAME + ".txt", "a")
    f.write(f"{testset}: acc = {acc:.4f}  ")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultroot")
    parser.add_argument("--modelstep", type=int, nargs="+")
    args = parser.parse_args()

    DATA_ROOT = 'D:\\Dataset\\evo'
    MODEL_ROOT = args.resultroot

    DEVICE = 'cuda'
    INPUT_SIZE = [112, 112]
    EMBEDDING_SIZE = config['EMBEDDING_SIZE']
    BATCH_SIZE = 512

    TXT_NAME = "Accuracy"

    if config['NET'] == 'r50':
        BACKBONE = get_model(config['NET'], dropout=0.0, fp16=config['FP16'], num_features=EMBEDDING_SIZE).cuda()

    # eval_list = [410000, 412000, 414000, 416000, 418000, 420000, 422000, 424000, 426000, 428000,
    #              430000, 432000, 434000, 436000, 438000, 440000, 442000, 444000, 446000, 448000,
    #              450000, 452000, 454000]
    eval_list = args.modelstep

    for i in eval_list:
        BACKBONE.load_state_dict(torch.load(MODEL_ROOT + f'\\model_step{i}.pt'))
        BACKBONE = BACKBONE.to(DEVICE)
        BACKBONE.eval()

        print(f"STEP: {i}")

        f = open(MODEL_ROOT + "\\" + TXT_NAME + ".txt", "a")
        f.write(f"Step: {i} :  ")
        f.close()

        Evaluate('lfw')
        Evaluate('calfw')
        Evaluate('cplfw')
        Evaluate('agedb_30')
        Evaluate('cfp_fp')
        Evaluate('vgg2_fp')

        f = open(MODEL_ROOT + "\\" + TXT_NAME + ".txt", "a")
        f.write("\n")
        f.close()
