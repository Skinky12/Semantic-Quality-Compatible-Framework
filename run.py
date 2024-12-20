import random
import os
import numpy as np
import scipy.io
from tqdm import tqdm
from dataloaders import DataLoaderIQA
from models.resnet_backbone import resnet50_backbone
from models.smrmI import SmRmSepNet
import torch
from scipy import stats
import argparse
import json
import scipy.io as scio
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import cv2


def SendEmail(msg_txt):
    smtpObj = smtplib.SMTP()
    smtpObj.connect('smtp.163.com', 25)
    smtpObj.login('maxiaoyucuc@163.com', 'YJAEFCSVZRDGOHSJ')
    msg = MIMEText('%s' % msg_txt, 'plain', 'utf-8')
    msg['From'] = Header('Xiaoyu Ma')
    msg['To'] = Header('Xiaoyu Ma')
    subject = 'Training End'
    msg['Subject'] = Header(subject, 'utf-8')
    smtpObj.sendmail('maxiaoyucuc@163.com', 'maxiaoyucuc@163.com', msg.as_string())


def getReuslts(len, type):
    plccs = 0.0
    srccs = 0.0
    for t in range(len):
        d1 = scipy.io.loadmat('./results/test_gt%s_cnt%06d.mat' % (type, t))['gt']
        d2 = scipy.io.loadmat('./results/test_pred%s_cnt%06d.mat' % (type, t))['pred']
        srcc_val_t, _ = stats.spearmanr(d1.squeeze(), d2.squeeze())
        plcc_val_t, _ = stats.pearsonr(d1.squeeze(), d2.squeeze())
        if plcc_val_t > plccs:
            plccs = plcc_val_t
        if srcc_val_t > srccs:
            srccs = srcc_val_t
    return plccs, srccs


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value


def DataSetup(root, batch_size, data_lens):
    scn_idxs = [x for x in range(data_lens[0])]
    np.random.shuffle(scn_idxs)
    scn_idxs_train = scn_idxs[:int(0.8 * data_lens[0])]
    scn_idxs_test = scn_idxs[int(0.8 * data_lens[0]):]

    loader_train = DataLoaderIQA('koniq', root, scn_idxs_train, batch_size=batch_size, istrain=True).get_data()
    loader_test = DataLoaderIQA('koniq', root, scn_idxs_test, batch_size=batch_size, istrain=False).get_data()
    return loader_train, loader_test


def test_model(models, loaders, config, cnt):
    torch.cuda.empty_cache()
    models.iqa.train(False)
    models.iqa.eval()
    my_device = torch.device('cuda:0')
    pred_vals = np.empty((0, 1))
    gt_vals = np.empty((0, 1))
    bcnt = 0
    for inputs, labels in loaders.test:
        inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
        img_ft = models.backbone(inputs)
        out = models.iqa(img_ft)
        pred = out['Q']

        pred_vals = np.append(pred_vals, pred.detach().cpu().numpy(), axis=0)
        gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)
        bcnt += 1

    scipy.io.savemat('./results/test_gt%s_cnt%06d.mat' % (config.type, cnt), {'gt': gt_vals})
    scipy.io.savemat('./results/test_pred%s_cnt%06d.mat' % (config.type, cnt), {'pred': pred_vals})

    srcc_val, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
    plcc_val, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

    models.iqa.train(True)
    return srcc_val, plcc_val


def train_model(models, loaders, optims, config):
    torch.cuda.empty_cache()
    models.iqa.train(True)
    my_device = torch.device('cuda:0')
    batch_cnt = 0
    for t in range(config.nepoch):
        pred_vals = np.empty((0, 1))
        gt_vals = np.empty((0, 1))
        epoch_loss = []

        for inputs, labels in tqdm(loaders.train):
            inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
            img_ft = models.backbone(inputs)
            out = models.iqa(img_ft)
            lossA = optims.criterion(out['Q'].squeeze(), labels.detach().squeeze())
            loss = lossA
            optims.optimA.zero_grad()
            loss.backward()
            optims.optimA.step()
            optims.schedA.step()

            epoch_loss.append(lossA.item())
            pred_vals = np.append(pred_vals, out['Q'].detach().cpu().numpy(), axis=0)
            gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

        print('testing....')
        srcc_val_t, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
        plcc_val_t, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

        srcc_val, plcc_val = test_model(models, loaders, config, t)
        print(
            'Test Phase: %05d SRCC : %.4f\t  PLCC : %.4f\t Train Phase=> SRCC: %.4f \t PLCC %.4f\t RecLoss: %.4f\t' % (
                t, srcc_val, plcc_val, srcc_val_t, plcc_val_t, sum(epoch_loss) / len(epoch_loss)))
        # torch.save(cur_model, './results/model.pkl')
        scipy.io.savemat('./results/train_gt%s_cnt%06d.mat' % (config.type, t), {'gt': gt_vals})
        scipy.io.savemat('./results/train_pred%s_cnt%06d.mat' % (config.type, t), {'pred': pred_vals})


def getHyperParams(sd):
    myconfigs = {
        'lrA': 2e-4,  # Up Block
        'lrB': 2e-4,  # Q Conv Block
        'lrC': 2e-4,  # Res Block
        'weight_decay': 5e-4,
        'T_MAX': 50,
        'eta_min': 0,
        'nepoch': 50,
        'batch_size': 12,
        'data_lens': (10073, 10073),
        'root': '/root/IQADatasets/koniq/',
        'type': 'KONIQ_SMRM__%04d' % sd
    }
    return Config(myconfigs)


def main(sdcfg):
    myseed = sdcfg.sd
    random.seed(myseed)
    os.environ['PYTHONHASHSEED'] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    myconfig = getHyperParams(myseed)

    mymodelbackbone = resnet50_backbone(pretrained=True).cuda()
    mymodelbackbone.train(False)
    mymodelbackbone.eval()

    mymodelIQA = SmRmSepNet().cuda()


    parasA = [{'params': mymodelIQA.squeeze1.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.squeeze2.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.squeeze3.parameters(), 'lr': myconfig['lrA']},
              #{'params': mymodelIQA.vsqueeze.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.ScaleMerge.parameters(), 'lr': myconfig['lrB']},
              {'params': mymodelIQA.SmQlMerge.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.qdense.parameters(), 'lr': myconfig['lrC']},
              ]

    optimizerA = torch.optim.Adam(parasA, weight_decay=myconfig.weight_decay)

    train_loader, test_loader = DataSetup(myconfig.root, myconfig.batch_size, myconfig.data_lens)
    schedulerA = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerA, myconfig.T_MAX, myconfig.eta_min)

    criterion = torch.nn.MSELoss()

    optim_params = Config({'criterion': criterion,
                           'optimA': optimizerA,
                           'schedA': schedulerA,
                           })
    models_params = Config({'backbone': mymodelbackbone,
                            'iqa': mymodelIQA,
                            })
    data_loaders = Config({'train': train_loader,
                           'test': test_loader})

    train_model(models_params, data_loaders, optim_params, myconfig)

    plss_max, srcc_max = getReuslts(myconfig.nepoch, myconfig.type)

    SendEmail('The Training of Type %s is Finished The Max PLCC is %0.4f | The Max SRCC is %0.4f' % (
        myconfig.type, plss_max, srcc_max))

    print('OK..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='sd', type=int, default=3407, help='Random Seed')
    sdcfg = parser.parse_args()
    main(sdcfg)


'''
import random
import os
import numpy as np
import scipy.io
from tqdm import tqdm
from dataloaders import DataLoaderIQA
from models.resnet_backbone import resnet50_backbone
from models.smrm import SmRmSepNet
import torch
from scipy import stats
import argparse
import json
import scipy.io as scio

import smtplib
from email.mime.text import MIMEText
from email.header import Header


def SendEmail(msg_txt):
    smtpObj = smtplib.SMTP()
    smtpObj.connect('smtp.163.com', 25)
    smtpObj.login('maxiaoyucuc@163.com', 'YJAEFCSVZRDGOHSJ')
    msg = MIMEText('%s' % msg_txt, 'plain', 'utf-8')
    msg['From'] = Header('Xiaoyu Ma')
    msg['To'] = Header('Xiaoyu Ma')
    subject = 'Training End'
    msg['Subject'] = Header(subject, 'utf-8')
    smtpObj.sendmail('maxiaoyucuc@163.com', 'maxiaoyucuc@163.com', msg.as_string())

def getReuslts(len, type):
    plccs = 0.0
    srccs = 0.0
    for t in range(len):
        d1 = scipy.io.loadmat('./results/test_gt%s_cnt%06d.mat' % (type, t))['gt']
        d2 = scipy.io.loadmat('./results/test_pred%s_cnt%06d.mat' % (type, t))['pred']
        srcc_val_t, _ = stats.spearmanr(d1.squeeze(), d2.squeeze())
        plcc_val_t, _ = stats.pearsonr(d1.squeeze(), d2.squeeze())
        if plcc_val_t > plccs:
            plccs = plcc_val_t
        if srcc_val_t > srccs:
            srccs = srcc_val_t
    return plccs, srccs


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value


def DataSetup(root, batch_size, data_lens):
    scn_idxs = [x for x in range(data_lens[0])]
    np.random.shuffle(scn_idxs)
    scn_idxs_train = scn_idxs[:int(0.8 * data_lens[0])]
    scn_idxs_test = scn_idxs[int(0.8 * data_lens[0]):]

    loader_train = DataLoaderIQA('koniq', root, scn_idxs_train, batch_size=batch_size, istrain=True).get_data()
    loader_test = DataLoaderIQA('koniq', root, scn_idxs_test, batch_size=batch_size, istrain=False).get_data()
    return loader_train, loader_test


def test_model(cur_model_back, cur_model_IQA, loader, cnt):
    torch.cuda.empty_cache()
    cur_model_IQA.train(False)
    cur_model_IQA.eval()
    my_device = torch.device('cuda:0')
    pred_vals = np.empty((0, 1))
    gt_vals = np.empty((0, 1))
    for inputs, labels in loader:
        inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
        ft = cur_model_back(inputs.detach())
        out = cur_model_IQA(ft)
        pred = out['Q']
        pred_vals = np.append(pred_vals, pred.detach().cpu().numpy(), axis=0)
        gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

    scipy.io.savemat('./results/test_gtZ_cnt%06d.mat' % (cnt), {'gt': gt_vals})
    scipy.io.savemat('./results/test_predZ_cnt%06d.mat' % (cnt), {'pred': pred_vals})

    srcc_val, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
    plcc_val, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

    cur_model_IQA.train(True)
    return srcc_val, plcc_val


def train_model(cur_model_back, cur_model_IQA, loader, test_loader, criterion, optimB, schedulerB, nepoch):
    torch.cuda.empty_cache()
    cur_model_IQA.train(True)
    my_device = torch.device('cuda:0')
    batch_cnt = 0
    for t in range(nepoch):
        pred_vals = np.empty((0, 1))
        gt_vals = np.empty((0, 1))
        epoch_loss = []

        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
            img_ft = cur_model_back(inputs)

            out = cur_model_IQA(img_ft)
            lossB = criterion(out['Q'].squeeze(), labels.detach().squeeze())

            optimB.zero_grad()
            lossB.backward()
            optimB.step()
            schedulerB.step()
            epoch_loss.append(lossB.item())

            batch_cnt += 1
            pred_vals = np.append(pred_vals, out['Q'].detach().cpu().numpy(), axis=0)
            gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

        print('testing....')
        srcc_val_t, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
        plcc_val_t, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

        srcc_val, plcc_val = test_model(cur_model_back, cur_model_IQA, test_loader, t)
        print(
            'Test Phase: %05d SRCC : %.4f\t  PLCC : %.4f\t Train Phase=> SRCC: %.4f \t PLCC %.4f\t ' % (
                t, srcc_val, plcc_val, srcc_val_t, plcc_val_t))
        # torch.save(cur_model, './results/model.pkl')
        scipy.io.savemat('./results/train_gtZ_cnt%06d.mat' % (t), {'gt': gt_vals})
        scipy.io.savemat('./results/train_predZ_cnt%06d.mat' % (t), {'pred': pred_vals})


def getHyperParams():
    myconfigs = {
        'lrA': 2e-4,  # Up Block
        'lrB': 2e-4,  # Q Conv Block
        'lrC': 2e-4,  # Res Block

        'weight_decay': 5e-4,
        'T_MAX': 50,
        'eta_min': 0,
        'nepoch': 20,
        'batch_size': 12,
        'data_lens': (10073, 10073),
        'root': 'E:\\ImageDatabase\\KONIQ\\',
    }
    return myconfigs


def main(cfg):

    myseed = 3407
    random.seed(myseed)
    os.environ['PYTHONHASHSEED'] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    myconfig = getHyperParams()

    mymodelbackbone = resnet50_backbone(pretrained=True).cuda()
    mymodelbackbone.train(False)
    mymodelbackbone.eval()

    mymodelIQA = SmRmSepNet().cuda()

    parasB = [{'params': mymodelIQA.squeeze1.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.squeeze2.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.squeeze3.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.vsqueeze.parameters(), 'lr': myconfig['lrB']},
              {'params': mymodelIQA.ScaleMerge.parameters(), 'lr': myconfig['lrB']},
              {'params': mymodelIQA.SmQlMerge.parameters(), 'lr': myconfig['lrA']},
              {'params': mymodelIQA.qdense.parameters(), 'lr': myconfig['lrC']},
              ]


    optimizerB = torch.optim.Adam(parasB, weight_decay=myconfig['weight_decay'])
    train_loader, test_loader = DataSetup(myconfig['root'], myconfig['batch_size'], myconfig['data_lens'])

    criterion = torch.nn.MSELoss()

    schedulerB = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerB, myconfig['T_MAX'], myconfig['eta_min'])
    train_model(mymodelbackbone, mymodelIQA, train_loader, test_loader, criterion, optimizerB, schedulerB,
                myconfig['nepoch'])

    print('OK..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    metacfg = parser.parse_args()
    main(metacfg)


'''
