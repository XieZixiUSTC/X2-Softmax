import torch
import os
import datetime
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import backbones
import loss
from configuration import config
from setup_seed import setup_seed

if __name__ == "__main__":
    default_cfg = [
        'ELASTICCOS',
        [64.0, 0.35, 0.05],
        ".\\OUTPUT\\202311 TEST",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", default=default_cfg[0])
    parser.add_argument("--lossparameters", default=default_cfg[1], type=float, nargs="+")
    parser.add_argument("--output", default=default_cfg[2])
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    cfg = config
    cfg['LOSS'] = args.loss
    cfg['LOSSPARAMETERS'] = args.lossparameters
    cfg['OUTPUT'] = args.output

    setup_seed(cfg['SEED'], cuda_deterministic=False)

    os.makedirs(cfg['OUTPUT'], exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_set = ImageFolder(cfg['TRAIN_ROOT'], train_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg['BATCH_SIZE'], shuffle=True,
                              pin_memory=True, num_workers=cfg['NUM_WORKERS'])
    cfg['NUM_IMG'] = len(train_set.imgs)
    cfg['NUM_CLASS'] = len(train_set.classes)

    backbone = backbones.get_model(cfg['NET'], dropout=0.0, fp16=cfg['FP16'],
                                   num_features=cfg['EMBEDDING_SIZE']).to(device)
    backbone.train()

    criterion, flag = loss.get_loss(cfg['LOSS'], cfg['LOSSPARAMETERS'])
    # 判断需不需要归一化
    if flag == 0:
        lastlayer = backbones.my_CE_0(criterion, cfg['EMBEDDING_SIZE'], cfg['NUM_CLASS'])
    elif flag == 1:
        lastlayer = backbones.my_CE_1(criterion, cfg['EMBEDDING_SIZE'], cfg['NUM_CLASS'])
    elif flag == 2:
        lastlayer = backbones.my_CE_2(criterion, cfg['EMBEDDING_SIZE'], cfg['NUM_CLASS'])
    elif flag == 3:
        lastlayer = backbones.MagLinear(criterion, cfg['EMBEDDING_SIZE'], cfg['NUM_CLASS'], cfg['LOSSPARAMETERS'])
    elif flag == 4:
        lastlayer = backbones.DynArcLinear(criterion, cfg['EMBEDDING_SIZE'], cfg['NUM_CLASS'])

    lastlayer.train().to(device)

    if cfg['OPTIMIZER'] == "SGD":
        optimizer = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": lastlayer.parameters()}],
            lr=cfg['LR'], momentum=cfg['MOMENTUM'], weight_decay=cfg['WEIGHT_DECAY']
        )

    cfg['STEPSPEREPOCH'] = cfg['NUM_IMG'] // cfg['BATCH_SIZE']
    for i in range(len(cfg['MILESTONES'])):
        cfg['MILESTONES'][i] = int(cfg['MILESTONES'][i] * cfg['STEPSPEREPOCH'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg['MILESTONES'],
        gamma=cfg['GAMMA'],
    )

    f = open(cfg['OUTPUT'] + "\\train_log.txt", "a")
    for key, value in cfg.items():
        num_space = 40 - len(key)
        text = key + " " * num_space + str(value)
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S : ") + text + "\n")
    f.close()

    step = 0
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    start_time = datetime.datetime.now()
    for epoch in range(0, cfg['NUM_EPOCH']):
        for _, (img, labels) in enumerate(train_loader):
            step += 1
            embeddings = backbone(img.to(device))
            loss: torch.Tensor = lastlayer(embeddings, labels.to(device))

            if cfg['FP16']:
                amp.scale(loss).backward()
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if step % 100 == 0:
                f = open(cfg['OUTPUT'] + "\\train_log.txt", "a")
                text = f"Loss:{float(loss):.4f}  Epoch:{epoch}  Step:{step:6d}  lr:{lr_scheduler.get_last_lr()[0]}"
                f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S : ") + text +
                        f" Left Time:{(datetime.datetime.now() - start_time) * (cfg['NUM_IMG'] * cfg['NUM_EPOCH'] / cfg['BATCH_SIZE'] - step) / 100}\n")
                f.close()
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S : ") + text
                      + f" Left Time:{(datetime.datetime.now() - start_time) * (cfg['NUM_IMG'] * cfg['NUM_EPOCH'] / cfg['BATCH_SIZE'] - step) / 100}")
                start_time = datetime.datetime.now()

                if (epoch >= cfg['NUM_EPOCH'] - 1) and (step % 2000 == 0):
                    path_module = os.path.join(cfg['OUTPUT'], f"model_step{step}.pt")
                    torch.save(backbone.state_dict(), path_module)
