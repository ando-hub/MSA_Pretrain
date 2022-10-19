import argparse
import os
import glob
import time
import copy
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.scheduler import CosineLRScheduler

from mytorch.logger import getlogger
from mytorch.dataset import MultimodalDataset, format_multimodal_data
from mytorch.scoring import get_score, get_total_score
from mytorch.writer import write_log
from mytorch.util import sec2hms, update_link
from mytorch.loss.multilabel_softmargin import MultiLabelSoftMarginLoss
from mytorch.optimizer.noam import NoamOpt
from mytorch.optimizer.util import get_optimizer_state, set_optimizer_state
from mytorch.model.multimodal_classifier import MultiModalClassifier
from mytorch.eval_performance import eval_performance
from mytorch.plot_modal_weight import plot_modal_weight


INPUT_MODAL = ['video', 'audio', 'text', 'videoaudio', 'videotext', 'audiotext', 'videoaudiotext']


def _parse():
    parser = argparse.ArgumentParser(description=u'train multimodal sentiment analysis')
    parser.add_argument('--model-save-dir', type=str, help='output model dir')
    parser.add_argument('--result-dir', type=str, help='output result file')
    parser.add_argument('--config-train', metavar='yaml',
                        help='training config: ./conf/train/*.yaml')
    parser.add_argument('--config-model', metavar='yaml',
                        help='model config: ./conf/model/*.yaml')
    parser.add_argument('--config-feat', metavar='yaml',
                        help='feature config: ./conf/feat/*.yaml')
    parser.add_argument('--gpu', type=int, default=-1,
                        help=u'GPU id (if <0 then use cpu)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help=u'number of workers in dataloader')
    # --- log config ---
    parser.add_argument('-l', metavar='logf', help='output log file')
    parser.add_argument('--loglevel',
                        choices=['error', 'warning', 'info', 'debug'],
                        default='info', help='output log level')
    # --- data config ---
    parser.add_argument('--video-data', type=str,
                        help='input video feature dir')
    parser.add_argument('--audio-data', type=str,
                        help='input audio feature dir')
    parser.add_argument('--text-data', type=str,
                        help='input text feature dir')
    # --- dataset/label config ---
    parser.add_argument('--trainset-list', type=str,
                        help='input trainset list')
    parser.add_argument('--validset-list', type=str,
                        help='input validationset list')
    parser.add_argument('--testset-list', type=str,
                        help='input testset list')
    # --- method config ---
    parser.add_argument('--input-modal',
                        choices=INPUT_MODAL,
                        default='videoaudiotext',
                        help=u'input modality: {}'.format(', '.join(INPUT_MODAL)))
    # --- fine-tuning option ---
    parser.add_argument('--init-model', type=str,
                        help=u'initial_model.pt')
    parser.add_argument('--init-model-dir', type=str,
                        help=u'initial_model dir')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help=u'finetune pretrained --init-model from 1st epoch')
    parser.add_argument('--test-only', action='store_true', default=False,
                        help=u'skip training and run inference')
    parser.add_argument('--attn-save-dir', type=str,
                        help='output attention dir (available when --test-only)')
    parser.add_argument('--embed-save-dir', type=str,
                        help='output embedding dir (available when --test-only)')
    return parser.parse_args()


def proc_1ep(
        dataloader, model, lossfunc=None, optimizer=None, device='cpu', mode='eval',
        result_file=None, attn_dir=None, embed_dir=None):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    if attn_dir:
        os.makedirs(attn_dir, exist_ok=True)
    if embed_dir:
        os.makedirs(embed_dir, exist_ok=True)
    if result_file:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        rslt_dic = defaultdict(list)

    accum_loss = []
    accum_score = []
    for x, t, uid in tqdm(dataloader, desc='[{}]'.format(mode)):
        # prepare input/output tensors
        x = [None if xi is None else xi.to(device) for xi in x]
        t = t.to(device)

        # decode
        if mode == 'train':
            optimizer.zero_grad()
            y, att, emb = model(x)
        else:
            with torch.no_grad():
                y, att, emb = model(x)
        if model.params['classification_type'] == 'regress':
            y = y.view(-1)

        # calculate loss
        if lossfunc:
            loss = lossfunc(y, t)
            if not torch.isnan(loss):
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                accum_loss.append(loss.item())

        # calculate score
        score = get_score(y, t, model.params['classification_type'])
        accum_score.append(score)

        # save inference results
        if result_file:
            rslt_dic['uid'].extend(uid)
            if model.params['classification_type'] == 'binary':
                prob = torch.sigmoid(y)
                pred = (prob > 0.5).int()
                for i in range(prob.shape[1]):
                    rslt_dic['actu_{}'.format(i)].extend(t[:, i].cpu().detach().tolist())
                    rslt_dic['pred_{}'.format(i)].extend(pred[:, i].cpu().detach().tolist())
                    rslt_dic['prob_{}'.format(i)].extend(prob[:, i].cpu().detach().tolist())
            elif model.params['classification_type'] == 'multiclass':
                prob = F.softmax(y, dim=1)
                pred = torch.argmax(prob, 1)
                rslt_dic['actu_0'].extend(t.cpu().detach().tolist())
                rslt_dic['pred_0'].extend(pred.cpu().detach().tolist())
                rslt_dic['prob_0'].extend(prob.cpu().detach().tolist())
            else:
                # raw output
                rslt_dic['ref_regress'].extend(t.cpu().detach().tolist())
                rslt_dic['out_regress'].extend(y.cpu().detach().tolist())
                # nonzero binary classification
                zero_idx = (t == 0).cpu().detach().numpy()
                nonzero_t = (t > 0).cpu().detach().numpy().astype(np.float32)
                nonzero_t[zero_idx] = np.nan
                nonzero_y = (y > 0).cpu().detach().numpy().astype(np.float32)
                nonzero_y[zero_idx] = np.nan
                rslt_dic['actu_nz2'].extend(nonzero_t.tolist())
                rslt_dic['pred_nz2'].extend(nonzero_y.tolist())
                # binary classification
                rslt_dic['actu_2'].extend((t >= 0).cpu().detach().tolist())
                rslt_dic['pred_2'].extend((y >= 0).cpu().detach().tolist())
                # 3/5/7-class classification
                rslt_dic['actu_3'].extend(t.cpu().detach().numpy().clip(-1, 1).round().tolist())
                rslt_dic['pred_3'].extend(y.cpu().detach().numpy().clip(-1, 1).round().tolist())
                rslt_dic['actu_5'].extend(t.cpu().detach().numpy().clip(-2, 2).round().tolist())
                rslt_dic['pred_5'].extend(y.cpu().detach().numpy().clip(-2, 2).round().tolist())
                rslt_dic['actu_7'].extend(t.cpu().detach().numpy().clip(-3, 3).round().tolist())
                rslt_dic['pred_7'].extend(y.cpu().detach().numpy().clip(-3, 3).round().tolist())

        # save attention outputs
        if attn_dir:
            att_v, att_a, att_t, att_dec = att
            for i, _uid in enumerate(uid):
                _dict = {}
                if att_v is not None:
                    _dict['att_video'] = att_v[i].cpu().detach().numpy()
                if att_a is not None:
                    _dict['att_audio'] = att_a[i].cpu().detach().numpy()
                if att_t is not None:
                    _dict['att_text'] = att_t[i].cpu().detach().numpy()
                if att_dec is not None:
                    _dict['att_dec'] = att_dec[i].cpu().detach().numpy()
                outf = os.path.join(attn_dir, _uid+'.npz')
                np.savez(outf, **_dict)

        # save embedding outputs
        if embed_dir:
            h_v, h_a, h_t = emb
            for i, _uid in enumerate(uid):
                _dict = {}
                if h_v is not None:
                    _dict['emb_video'] = h_v[i].cpu().detach().numpy()
                if h_a is not None:
                    _dict['emb_audio'] = h_a[i].cpu().detach().numpy()
                if h_t is not None:
                    _dict['emb_text'] = h_t[i].cpu().detach().numpy()
                outf = os.path.join(embed_dir, _uid+'.npz')
                np.savez(outf, **_dict)

    # write rslt_dic to result_file (csv format)
    if result_file:
        df = pd.DataFrame.from_dict(rslt_dic)
        df = df.sort_index(axis=1)
        df = df.sort_values(by='uid')
        df.to_csv(result_file, index=False)

    avg_loss = sum(accum_loss)/len(accum_loss) if accum_loss else float('inf')
    total_score = get_total_score(accum_score) if accum_score else {}
    return model, avg_loss, total_score


def _main(args):
    logger = getlogger(args.l, args.loglevel)
    if args.model_save_dir:
        os.makedirs(args.model_save_dir, exist_ok=True)
    if args.result_dir:
        os.makedirs(args.result_dir, exist_ok=True)

    # load training / inference parameters
    logger.info('load train config: {}'.format(args.config_train))
    with open(args.config_train) as fp:
        train_param = yaml.safe_load(fp)

    logger.info('load feat config: {}'.format(args.config_feat))
    with open(args.config_feat) as fp:
        train_param.update(yaml.safe_load(fp))

    # load model parameters
    logger.info('load model config: {}'.format(args.config_model))
    if args.init_model or args.init_model_dir:
        if args.init_model_dir:
            _models = glob.glob(os.path.join(args.init_model_dir, 'model.*.pt'))
            _models = list(set(_models) - {'model.pt'})
            init_model = sorted(_models)[-1]
        else:
            init_model = args.init_model
        load_data = torch.load(init_model)
        model_param = load_data['model_param']
    else:
        init_model = None
        with open(args.config_model) as fp:
            model_param = yaml.safe_load(fp)

    logger.info('prepare device ...')
    torch.manual_seed(train_param['seed'])
    if args.gpu >= 0:
        device = 'cuda:{}'.format(args.gpu)
        torch.cuda.manual_seed(train_param['seed'])
        torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'

    logger.info('load dataset ...')
    if args.trainset_list:
        trn_dataset = MultimodalDataset(
                args.trainset_list,
                args.video_data, args.audio_data, args.text_data,
                args.input_modal,
                train_param['video_feat'],
                train_param['audio_feat'],
                train_param['text_feat'],
                iseval=False,
                )
        trn_loader = DataLoader(
                trn_dataset, collate_fn=format_multimodal_data,
                batch_size=train_param['batch_size'],
                drop_last=True,
                shuffle=True,
                num_workers=args.num_workers,
                timeout=10,
                )
    else:
        trn_loader = None
    if args.validset_list:
        dev_dataset = MultimodalDataset(
                args.validset_list,
                args.video_data, args.audio_data, args.text_data,
                args.input_modal,
                train_param['video_feat'],
                train_param['audio_feat'],
                train_param['text_feat'],
                iseval=True
                )
        dev_loader = DataLoader(
                dev_dataset, collate_fn=format_multimodal_data,
                batch_size=train_param['batch_size'],
                drop_last=False,
                shuffle=False,
                num_workers=args.num_workers,
                timeout=10,
                )
    else:
        dev_loader = None
    if args.testset_list:
        tst_dataset = MultimodalDataset(
                args.testset_list,
                args.video_data, args.audio_data, args.text_data,
                args.input_modal,
                train_param['video_feat'],
                train_param['audio_feat'],
                train_param['text_feat'],
                iseval=True
                )
        tst_loader = DataLoader(
                tst_dataset, collate_fn=format_multimodal_data,
                batch_size=train_param['batch_size'],
                drop_last=False,
                shuffle=False,
                num_workers=args.num_workers,
                timeout=10,
                )
    else:
        tst_loader = None

    if not init_model:
        logger.info('get input/output sizes and class-weights from dataset ...')
        model_param['input_dims'] = trn_dataset.get_input_dims()
        model_param['input_layers'] = trn_dataset.get_input_layers()
        model_param['output_dim'], model_param['classification_type'] = trn_dataset.get_output_dims()
    if trn_loader:
        if 'class_weight' not in train_param or train_param['class_weight']:
            class_weight = trn_dataset.get_class_weight()
        else:
            class_weight = None

    logger.info('initialize model ...')
    model = MultiModalClassifier(model_param)
    if init_model:
        model.load_state_dict(load_data['model'])
    model.to(device)
    logger.debug('model structure:\n{}'.format(model))

    # training
    if not args.test_only:
        logger.info('prepare optimizer ...')
        param_set = filter(lambda p: p.requires_grad, model.parameters())
        if train_param['optimizer'] == 'adam':
            optimizer = optim.Adam(param_set, lr=train_param['lr'])
        elif train_param['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_set, lr=train_param['lr'])
        elif train_param['optimizer'] == 'radam':
            optimizer = optim.RAdam(param_set, lr=train_param['lr'])
        elif train_param['optimizer'] == 'noam':
            optimizer = NoamOpt(args.hidden_size, args.noam_scale, args.noam_warmup_steps,
                                torch.optim.Adam(param_set, lr=0, betas=(0.9, 0.98), eps=1e-9))
        else:
            raise ValueError(train_param['optimizer'])

        if 'scheduler' in train_param and train_param['scheduler']:
            scheduler = CosineLRScheduler(
                    optimizer, t_initial=train_param['max_epoch'],
                    lr_min=train_param['lr']*0.1, warmup_t=int(train_param['max_epoch']/5),
                    warmup_lr_init=train_param['lr']*0.1, warmup_prefix=True
                    )
        else:
            scheduler = None

        logger.info('prepare lossfunc ...')
        if model_param['classification_type'] == 'binary':
            if class_weight:
                w = torch.tensor(np.stack(class_weight).T, dtype=torch.float32).to(device)
            else:
                w = None
            lossfunc = MultiLabelSoftMarginLoss(weight=w)
        elif model_param['classification_type'] == 'multiclass':
            if class_weight:
                w = torch.tensor(class_weight, dtype=torch.float32).to(device)
            else:
                w = None
            lossfunc = nn.CrossEntropyLoss(weight=w)
        elif model_param['classification_type'] == 'regress':
            lossfunc = nn.L1Loss()
        else:
            raise ValueError('invalid classification_type: {}'.format(model_param['classification_type']))

        logger.info('prepare training logger ...')
        log_base = os.path.join(args.model_save_dir, 'log', datetime.now().strftime('%Y%m%d.%H%M%S'))
        writer_trn = SummaryWriter(log_dir=os.path.join(log_base, 'train'))
        writer_dev = SummaryWriter(log_dir=os.path.join(log_base, 'dev'))
        writer_tst = SummaryWriter(log_dir=os.path.join(log_base, 'test'))

        best_modelf = os.path.join(args.model_save_dir, 'model.pt')
        best_dev_resultf = os.path.join(args.result_dir, 'result.dev.csv')
        best_dev_scoref = os.path.join(args.result_dir, 'result.dev.txt')
        best_tst_resultf = os.path.join(args.result_dir, 'result.tst.csv')
        best_tst_scoref = os.path.join(args.result_dir, 'result.tst.txt')

        logger.info('start training ...')
        start_time = time.time()
        _best = {
                'epoch': -1,
                'loss': float('inf'),
                'model': None,
                'optimizer': None,
                'scheduler': None,
                'model_param': model_param,
                }
        _now = copy.deepcopy(_best)
        if init_model and not args.finetune:
            _best['epoch'] = load_data['epoch']
            _best['loss'] = load_data['loss']
            _best['model'] = load_data['model']
            _best['optimizer'] = load_data['optimizer']
            _best['scheduler'] = load_data['scheduler']
            set_optimizer_state(optimizer, load_data['optimizer'], device)
        for ep in range(1, train_param['max_epoch']+1):
            if ep <= _best['epoch']:
                logger.info('Ep: {}/{} skip training (resume)'.format(ep, train_param['max_epoch']))
                continue

            # run 1 epoch
            dev_resultf = os.path.join(args.result_dir, 'result.dev.{:04d}.csv'.format(ep))
            tst_resultf = os.path.join(args.result_dir, 'result.tst.{:04d}.csv'.format(ep))
            dev_scoref = os.path.join(args.result_dir, 'result.dev.{:04d}.txt'.format(ep))
            tst_scoref = os.path.join(args.result_dir, 'result.tst.{:04d}.txt'.format(ep))
            model, trn_loss, trn_score = proc_1ep(
                    trn_loader, model, lossfunc, optimizer, device, mode='train',
                    )
            _, dev_loss, dev_score = proc_1ep(
                    dev_loader, model, lossfunc, optimizer, device, mode='eval',
                    result_file=dev_resultf
                    )
            _, tst_loss, tst_score = proc_1ep(
                    tst_loader, model, lossfunc, optimizer, device, mode='eval',
                    result_file=tst_resultf
                    )
            eval_performance(dev_resultf, dev_scoref)
            eval_performance(tst_resultf, tst_scoref)

            # early-stopping
            if dev_loss <= _best['loss'] or ep % 5 == 0 or ep == train_param['max_epoch']:
                logger.info('Ep: {} save model ...'.format(ep))
                modelf = os.path.join(args.model_save_dir, 'model.{:04d}.pt'.format(ep))
                model.to('cpu')
                _now['epoch'] = ep
                _now['loss'] = dev_loss
                _now['model'] = copy.deepcopy(model.state_dict())
                _now['optimizer'] = get_optimizer_state(optimizer)
                _now['scheduler'] = scheduler
                torch.save(_now, modelf)
                model.to(device)

                # link the best model, result, ...
                if dev_loss <= _best['loss']:
                    _best = copy.deepcopy(_now)
                    update_link(modelf, best_modelf)
                    update_link(dev_resultf, best_dev_resultf)
                    update_link(dev_scoref, best_dev_scoref)
                    update_link(tst_resultf, best_tst_resultf)
                    update_link(tst_scoref, best_tst_scoref)

            # update scheduler
            if scheduler is not None:
                scheduler.step(ep)

            # save log
            write_log(writer_trn, ep, {'base/loss': trn_loss, **trn_score})
            write_log(writer_dev, ep, {'base/loss': dev_loss, **dev_score})
            write_log(writer_tst, ep, {'base/loss': tst_loss, **tst_score})

            # show rest time
            elapsed_time = time.time() - start_time
            rest_time = elapsed_time*(train_param['max_epoch']-ep)/ep
            logger.info('Ep: {}/{} [loss] trn={:.3f}, dev={:.3f} tst={:.3f}'.format(
                ep, train_param['max_epoch'],
                trn_loss, dev_loss, tst_loss
                ))
            logger.info('Ep: {}/{} [acc]  trn={:.3f}, dev={:.3f} tst={:.3f} (proc:{}, rest:{})'.format(
                ep, train_param['max_epoch'],
                trn_score['score/accuracy_total'],
                dev_score['score/accuracy_total'],
                tst_score['score/accuracy_total'],
                sec2hms(elapsed_time), sec2hms(rest_time)
                ))

        logger.info('best: Ep: {} (dev loss={:.3f})'.format(_best['epoch'], _best['loss']))
        writer_trn.close()
        writer_dev.close()
        writer_tst.close()
        logger.info('Finished training!')
    else:
        logger.info('Inference ...')
        resultf = os.path.join(args.result_dir, 'result.csv')
        attn_dir = os.path.join(args.attn_save_dir, 'model')
        embed_dir = os.path.join(args.embed_save_dir, 'model')
        _, _, tst_score = proc_1ep(
                tst_loader, model, device=device, mode='eval',
                result_file=resultf, attn_dir=attn_dir, embed_dir=embed_dir
                )
        if resultf:
            scoref = os.path.join(args.result_dir, 'result.txt')
            eval_performance(resultf, scoref)
        if attn_dir:
            attn_png = os.path.join(args.attn_save_dir, 'model.png')
            plot_modal_weight(attn_dir, attn_png)
    return


if __name__ == '__main__':
    args = _parse()
    _main(args)
