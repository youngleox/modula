import sys
import os
import signal
import math
import torch
import numpy
import argparse
import pickle

from tqdm.auto import trange
from data.dataset import getIterator
from module.compound import MLP, ResMLP, CNN, ResCNN

architectures = ['mlp', 'resmlp', 'cnn', 'rescnn']
datasets      = ['cifar10']
losses        = ['mse', 'xent']
optims        = ['mgd', 'adamw']

parser = argparse.ArgumentParser()

# system
parser.add_argument('--cpu',            action='store_true'      )
parser.add_argument('--log_dir',        type=str,   default='logs/temp')
parser.add_argument('--log_interval',   type=int,   default=100  )
parser.add_argument('--seed',           type=int,   default=0    )
parser.add_argument('--batch_size',     type=int,   default=128  )
parser.add_argument('--train_steps',    type=int,   default=1000 )
parser.add_argument('--test_steps',     type=int,   default=100  )
parser.add_argument('--dataset',        type=str,   default='cifar10',  choices=datasets)

# architecture
parser.add_argument('--arch',       type=str,   default='mlp',      choices=architectures)
parser.add_argument('--depth',      type=int,   default=6    )
parser.add_argument('--blockdepth', type=int,   default=2    )
parser.add_argument('--width',      type=int,   default=384  )

# training
parser.add_argument('--optim',      type=str,   default='mgd',      choices=optims)
parser.add_argument('--loss',       type=str,   default='xent',     choices=losses)
parser.add_argument('--lr',         type=float, default=0.5  )
parser.add_argument('--beta',       type=float, default=0.9  )
parser.add_argument('--wd',         type=float, default=0.01 )

def evalute(output, data, target):

    acc = (output.argmax(dim=1) == target).sum() / target.numel()

    if args.loss == 'mse':
        onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
        error = (output - onehot * math.sqrt(output.shape[1])).square().mean(dim=1)

    elif args.loss == 'xent':
        error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)

    loss = error.mean()

    return loss, acc


if __name__ == '__main__':

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    pickle.dump(vars(args), open( os.path.join(args.log_dir, 'args.pickle'), "wb" ) )
    for arg in vars(args):
        print("{: <20} {: <20}".format(arg, getattr(args, arg)))

    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    _getBatch, input_dim, output_dim = getIterator(dataset="cifar10", batch_size=args.batch_size)

    def cleanup(sig=None, frame=None):
        global _getBatch
        del _getBatch
        print(" Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    def getBatch(train):
        data, target = _getBatch(train)
        if not args.cpu: data, target = data.cuda(), target.cuda()
        if 'mlp' in args.arch: data = data.flatten(start_dim=1)
        return data, target

    if args.arch == "resmlp":
        net = ResMLP(   width = args.width,
                        num_blocks = args.depth,
                        block_depth = args.blockdepth,
                        input_dim = numpy.prod(input_dim),
                        output_dim = output_dim
                    )

    elif args.arch == "mlp":
        net = MLP(      width = args.width,
                        depth = args.depth,
                        input_dim = numpy.prod(input_dim),
                        output_dim = output_dim
                    )

    elif args.arch == "cnn":
        net = CNN(      width = args.width,
                        depth = args.depth,
                        input_dim = input_dim[0],
                        output_dim = output_dim,
                        num_pixels = input_dim[1]*input_dim[2]
                    )

    elif args.arch == "rescnn":
        net = ResCNN(   width = args.width,
                        num_blocks = args.depth,
                        block_depth = args.blockdepth,
                        input_dim = input_dim[0],
                        output_dim = output_dim,
                        num_pixels = input_dim[1]*input_dim[2]
                    )

    print(net)

    net.initialize(device = "cpu" if args.cpu else "cuda")

    if args.optim == "adamw":
        optim = torch.optim.AdamW(net.parameters, lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.wd)

    results = {"train_loss":[], "test_loss":[], "train_acc":[], "test_acc":[]}

    for step in (pbar := trange(args.train_steps + 1, file=sys.stdout)):

        if step % args.log_interval == 0:
            test_loss = test_acc = 0
            for _ in range(args.test_steps):
                data, target = getBatch(train = False)
                with torch.no_grad(): loss, acc = evalute(net.forward(data), data, target)

                test_loss += loss
                test_acc += acc

            results["test_loss"].append(test_loss.item() / args.test_steps)
            results["test_acc"].append(test_acc.item() / args.test_steps)

        data, target = getBatch(train = True)
        train_loss, train_acc = evalute(net.forward(data), data, target)

        train_loss.backward()

        schedule = 1 - step / args.train_steps
        if args.optim == "mgd":
            net.update(lr=args.lr*schedule, beta=args.beta, wd=args.wd)
        else:
            for g in optim.param_groups: g['lr'] = args.lr*schedule
            optim.step()
            optim.zero_grad()

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())

        if step % args.log_interval == 0:
            pickle.dump(results, open( os.path.join(args.log_dir, 'results.pickle'), "wb" ) )
            pbar.set_description(f"train: {numpy.mean(results['train_acc'][-100:]):.4f} // test: {results['test_acc'][-1]:.4f}")

            if step > 0 and math.isnan(train_loss): break

    cleanup()
