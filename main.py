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
from module.compound import *

architectures = ['resmlp', 'rescnn', 'gpt']
datasets      = ['cifar10', 'shakespeare']
losses        = ['mse', 'xent']

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
parser.add_argument('--arch',           type=str,   default='resmlp',   choices=architectures)
parser.add_argument('--depth',          type=int,   default=6    )
parser.add_argument('--block_depth',    type=int,   default=2    )
parser.add_argument('--width',          type=int,   default=384  )
parser.add_argument('--context',        type=int,   default=256  )
parser.add_argument('--num_heads',      type=int,   default=8    )
parser.add_argument('--d_embed',        type=int,   default=128  )
parser.add_argument('--d_query',        type=int,   default=16   )
parser.add_argument('--d_value',        type=int,   default=16   )

# training
parser.add_argument('--normalize',      action='store_true'      )
parser.add_argument('--loss',           type=str,   default='xent',     choices=losses)
parser.add_argument('--lr',             type=float, default=0.5  )
parser.add_argument('--beta1',          type=float, default=0.9  )
parser.add_argument('--beta2',          type=float, default=0.99 )
parser.add_argument('--wd',             type=float, default=0.01 )

def evalute(output, data, target):

    if args.arch == "gpt":
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

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

    getBatch, input_dim, output_dim = getIterator(  dataset = args.dataset,
                                                    batch_size = args.batch_size,
                                                    context = args.context,
                                                    device = "cpu" if args.cpu else "cuda" )

    def cleanup(sig=None, frame=None):
        global getBatch
        del getBatch
        print("Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    if args.arch == "resmlp":
        net = ResMLP(args.width, args.depth, args.block_depth, input_dim, output_dim)

    elif args.arch == "rescnn":
        net = ResCNN(args.width, args.depth, args.block_depth, input_dim, output_dim)

    elif args.arch == "gpt":
        net = GPT(  vocab_size = input_dim,
                    context = args.context,
                    num_heads = args.num_heads,
                    d_embed = args.d_embed,
                    d_query = args.d_query,
                    d_value = args.d_value,
                    num_blocks = args.depth )

    print(net)

    weights = net.initialize(device = "cpu" if args.cpu else "cuda")
    mom1 = 0 * weights
    if args.beta2 >= 0:
        mom2 = 0 * weights

    results = {"train_loss":[], "test_loss":[], "train_acc":[], "test_acc":[]}

    for step in (pbar := trange(args.train_steps + 1, file=sys.stdout)):

        if step % args.log_interval == 0:
            test_loss = test_acc = 0
            for _ in range(args.test_steps):
                data, target = getBatch(train = False)
                with torch.no_grad(): loss, acc = evalute(net.forward(data, weights), data, target)

                test_loss += loss
                test_acc += acc

            results["test_loss"].append(test_loss.item() / args.test_steps)
            results["test_acc"].append(test_acc.item() / args.test_steps)

        data, target = getBatch(train = True)
        train_loss, train_acc = evalute(net.forward(data, weights), data, target)

        train_loss.backward()

        schedule = 1 - step / args.train_steps

        with torch.no_grad():
            mom1 += (1-args.beta1)**(step/(step+1)) * (weights.grad()    - mom1)

            update = mom1

            if args.beta2 >= 0:
                mom2 += (1-args.beta2)**(step/(step+1)) * (weights.grad()**2 - mom2)
                update = update / mom2 ** 0.5

            if args.normalize:
                update = net.normalize(update)

            weights -= args.lr * schedule * update
            weights -= args.lr * schedule * args.wd * weights

            weights.zero_grad()

        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc.item())

        if step % args.log_interval == 0:
            pickle.dump(results, open( os.path.join(args.log_dir, 'results.pickle'), "wb" ) )
            pbar.set_description(f"train: {numpy.mean(results['train_acc'][-100:]):.4f} // test: {results['test_acc'][-1]:.4f}")

            if step > 0 and math.isnan(train_loss): break

    cleanup()
