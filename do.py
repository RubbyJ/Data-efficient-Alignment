#!/usr/bin/python3
# -*- coding: utf-8 -*
import argparse
import random

from torch.utils.tensorboard import SummaryWriter

from src.dataset.NeuMatch_YMS_BERT_Dataset import NeuMatch_YMS_BERT_Dataset
from src.dataset.NeuMatch_YMS_LXMERT_Dataset import NeuMatch_YMS_LXMERT_Dataset
from src.model.AlignNet import AlignNet
from src.engine.train import *
from src.engine.evaluate import *
from src.solver.warm_up import linear_warm_up
from src.utils.LabelSmoothing import LabelSmoothingLoss


def main():
    """ main function """

    """ distributed """
    args.distributed = False
    args.world_size = 1
    torch.cuda.set_device(0)

    """ model init """
    AlignNet_instance = AlignNet(args).cuda()

    if args.loss == 'ce':
        print('** Using Cross Entropy Loss **')
        criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    elif args.loss == 'ls':
        print('** Using Label Smoothing Regularization Loss **')
        criterion = LabelSmoothingLoss(reduction='none', classes=args.actions_nc,
                                       smoothing=args.lsr_epsilon).cuda()
    else:
        raise NotImplementedError

    parameters = [{'params': p} for p in AlignNet_instance.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_decay == 'OnPlateau':
        print('Do OnPlateau Lr Decay with {} patience, {}x factor!'.format(args.patience, args.factor))
        scheduler_OP = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.factor,
                                                                  patience=args.patience,
                                                                  verbose=True, threshold_mode='rel')
    else:
        raise NotImplementedError

    """ evaluation """
    if args.evaluate:
        if args.dataset == 'yms':
            if args.LXMERT:
                test_db = NeuMatch_YMS_LXMERT_Dataset('../data/', args.dataset, args.max_seq_len, 'test')
            else:
                test_db = NeuMatch_YMS_BERT_Dataset('../data/', args.dataset, args.max_seq_len, 'test', args.random_project)
        else:
            raise ValueError

        test_dataLoader = torch.utils.data.DataLoader(test_db, batch_size=args.batch_size,
                                                      shuffle=False, num_workers=0,
                                                      pin_memory=True)
        ckpt = torch.load(os.path.join(
            './checkpoints', args.where_best,
            args.where_best_epoch
        ))['model_state_dict']

        AlignNet_instance.load_state_dict(ckpt)

        evaluate(test_dataLoader, AlignNet_instance, args, None, None, phase='Test')
        return

    """ data """
    if args.dataset == 'yms':
        print("* YMS dataset *")
        if args.LXMERT:
            print("** LXMERT FEATURE **")
            train_db = NeuMatch_YMS_LXMERT_Dataset('../data/', args.dataset, args.training_max_seq_len, 'training')
            val_db = NeuMatch_YMS_LXMERT_Dataset('../data/', args.dataset, args.max_seq_len, 'val')
            test_db = NeuMatch_YMS_LXMERT_Dataset('../data/', args.dataset, args.max_seq_len, 'test')
        else:
            train_db = NeuMatch_YMS_BERT_Dataset('../data/', args.dataset, args.training_max_seq_len,
                                                     'training', args.random_project)
            val_db = NeuMatch_YMS_BERT_Dataset('../data/', args.dataset, args.max_seq_len, 'val',
                                                   args.random_project)
            test_db = NeuMatch_YMS_BERT_Dataset('../data/', args.dataset, args.max_seq_len, 'test',
                                                    args.random_project)
    else:
        raise ValueError

    train_sampler = None
    val_sampler = None
    test_sampler = None

    train_dataLoader = torch.utils.data.DataLoader(train_db, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_dataLoader = torch.utils.data.DataLoader(val_db, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=0,
                                                 sampler=val_sampler, pin_memory=True)
    test_dataLoader = torch.utils.data.DataLoader(test_db, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0,
                                                  sampler=test_sampler, pin_memory=True)
    print('# train db: %d, val db: %d' % (len(train_db), len(val_db)))

    """ train & validate """
    if args.warm_up_length > 0:
        print('DO {}-epochs WARM-UP !'.format(args.warm_up_length))
        args.warm_up = True
    else:
        args.warm_up = False

    clip_obj = 0.05
    sent_obj = 0.01

    for epoch in range(args.epochs):
        """ Warm-Up """
        if args.warm_up:
            linear_warm_up(optimizer, epoch, args.lr, args.warm_up_length)

        """ train """
        train_loss = train(
            train_dataLoader,
            AlignNet_instance, 
            criterion,
            optimizer,
            epoch, args, writer
        )

        if args.lr_decay == 'OnPlateau' and epoch >= args.warm_up_length:
            scheduler_OP.step(train_loss)

        """ Printing Validation """
        if (epoch + 1) % args.val_freq == 0:
            clip_acc, sent_acc = evaluate(val_dataLoader, AlignNet_instance, args, writer, epoch, phase='Validate')

            if clip_acc > clip_obj:
                torch.save({'model_state_dict': AlignNet_instance.state_dict(), 'epoch': epoch + 1},
                           os.path.join(args.ckt_root, args.name, 'best_val_clip_acc.ckpt'))
                clip_obj = clip_acc

            if sent_acc > sent_obj:
                torch.save({'model_state_dict': AlignNet_instance.state_dict(), 'epoch': epoch + 1},
                           os.path.join(args.ckt_root, args.name, 'best_val_sent_iou.ckpt'))
                sent_obj = sent_acc
    
        """ Printing Test """
        if (epoch + 1) % args.test_freq == 0:
            evaluate(test_dataLoader, AlignNet_instance, args, writer, epoch, phase='Test')

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': AlignNet_instance.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.ckt_root, args.name, 'NeuMATCHING_%d.ckpt' % (epoch + 1)))

            print("Successfully store the model in %s" %
                  (os.path.join(args.ckt_root, args.name, 'NeuMATCHING_%d.ckpt' % (epoch + 1))))


if __name__ == '__main__':
    # global args
    parser = argparse.ArgumentParser(description='NeuMARCHING Training')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='only evaluate the model')
    parser.add_argument('--seed', default=101, type=int, metavar='N',
                        help='random seed (default:101)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Num workers for dataLoader')

    parser.add_argument("--dataset", type=str, default="yms",
                        help="the dataset name")
    parser.add_argument('--LXMERT', action='store_true',
                        help='To use LXMERT feature')
    parser.add_argument("--ckt_root", type=str, default="./checkpoints",
                        help="the folder path of checkpoints")

    parser.add_argument('--where_best', type=str,
                        default=None,
                        help='where to find the best ckpt')
    parser.add_argument('--where_best_epoch', type=str, default='NeuMATCHING_350.ckpt',
                        help='For Loading Model')

    """ Regularization """
    parser.add_argument('--dropout', default=0., type=float,
                        help='dropout on Classification FCs')
    parser.add_argument('--rnn_dropout', default=0., type=float,
                        help='dropout on RNN')
    parser.add_argument('--encoder_dropout', default=0.0, type=float,
                        help='dropout on encoder fully connected layers')
    parser.add_argument('--SBN', action='store_true',
                        help='To do Sequence-wise Batch Normalization During Video&Text Stack')
    parser.add_argument('--LN_num', default=None, type=int,
                        help='number of Layer Normalization with Stacks')
    parser.add_argument('--loss', default='ce', type=str,
                        help='\'ce\' for cross entropy, \'ls\' for adding label smoothing.')
    parser.add_argument('--lsr_epsilon', default=0.03, type=float,
                        help='Label Smoothing regularization parameter')

    """ Optimization """
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument("--epochs", type=int, default=350,
                        help='number of total epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--adamlars', action='store_true',
                        help='AdamLARS')
    parser.add_argument('--lars_coef', default=0.001, type=float,
                        help='LARS trust coefficient')
    parser.add_argument('--lr_decay', default='OnPlateau', type=str,
                        help='lr decay supports OnPlateau')
    parser.add_argument('--warm_up_length', default=0, type=int,
                        help='How long for Warm Up')
    parser.add_argument('--patience', default=10, type=int,
                        help='How long to perform the patience when OnPlateau')
    parser.add_argument('--factor', default=0.5, type=float,
                        help='The factor to multiple with lr when OnPlateau')

    """ Model Architecture """
    parser.add_argument('--actions_nc', default=3, type=int,
                        help='the types of action')
    parser.add_argument('--arnn_size', default=8, type=int,
                        help='hidden dimensions for the action stack')
    parser.add_argument('--mrnn_size', default=20, type=int,
                        help='hidden dimensions for the matched stack')
    parser.add_argument('--max_video_len', default=249)
    parser.add_argument('--max_sent_len', default=95)
    parser.add_argument('--max_act_len', default=317)
    parser.add_argument('--teacher_forcing', default=1.0, type=float, metavar='N',
                        help='teacher forcing (default:1.0)')
    parser.add_argument('--random_project', action='store_true', default=False,
                        help='To random project Video/Text feature or not')

    """ Logger """
    parser.add_argument('--val_freq', default=1, type=int, metavar='N',
                        help='val frequency')
    parser.add_argument('--test_freq', default=1, type=int, metavar='N',
                        help='test frequency')
    parser.add_argument('--save_freq', '-s', default=100, type=int, metavar='N',
                        help='save frequency')

    args = parser.parse_args()

    timestr = time.strftime("%m.%d-%H:%M")

    args.training_max_seq_len = (94, 40, 108)
    args.max_seq_len = (args.max_video_len, args.max_sent_len, args.max_act_len)  # For Val & Test

    if args.dataset == 'yms':
        """ We index all the YMS data together. """
        args.duration = np.load(os.path.join('../data', args.dataset+'_origin_pixel_dimnorm',
                                             '{}_segment_duration.npy'.format(args.dataset)),
                                allow_pickle=True).item()  # The Duration data for computing sent. IoU
        # args.training_duration = args.val_duration = args.test_duration = args.duration

        args.matched_gt = np.load(os.path.join('../data', args.dataset+'_origin_pixel_dimnorm',
                                               '{}_sent_ground_truth.npy'.format(args.dataset)),
                                  allow_pickle=True).item()
        # args.training_gt = args.val_gt = args.test_gt = args.gt

        args.unmatched_gt = np.load(os.path.join('../data', args.dataset+'_origin_pixel_dimnorm',
                                                 '{}_unmatched_ground_truth.npy'.format(args.dataset)),
                                    allow_pickle=True)
    else:
        raise ValueError

    args.name = timestr + \
        str(args.dataset) + \
        'RP' * args.random_project + 'Full' * (1 - args.random_project) + \
        'SBN' * args.SBN + 'LN{}'.format(args.LN_num) * (args.LN_num is not None) + \
        ('LSR' + str(args.lsr_epsilon)) * (args.loss == 'ls') + \
        'WarmUp' + str(args.warm_up_length) + \
        ('AdamLARS' + '_coef' + str(args.lars_coef)) * args.adamlars

    if not args.evaluate:
        os.makedirs(os.path.join(args.ckt_root, args.name), exist_ok=True)

        writer = SummaryWriter('./{}_logs/'.format(args.dataset) + args.name)

    """ Deterministic """
    torch.backends.cudnn.benchmark = True
    # deterministic
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    with torch.autograd.set_detect_anomaly(True):
        torch.cuda.synchronize()
        start = time.time()

        main()

        torch.cuda.synchronize()
        end = time.time()
        print('Total time: {} mins.'.format((end - start) / 60))
        print("Memory allocated:{:.2f}G".format(
            torch.cuda.max_memory_allocated() /
            1024 ** 3))

    if not args.evaluate:
        writer.close()
