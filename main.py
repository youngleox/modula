import math
import torch
import numpy
import argparse

from tqdm.auto import trange
from data.dataset import getIterator
from module.compound import MLP, ResMLP

architectures = ['mlp', 'resmlp']
datasets      = ['cifar10']
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
parser.add_argument('--arch',       type=str,   default='mlp',      choices=architectures)
parser.add_argument('--depth',      type=int,   default=6    )
parser.add_argument('--blockdepth', type=int,   default=2    )
parser.add_argument('--width',      type=int,   default=384  )

# training
parser.add_argument('--loss',       type=str,   default='xent',     choices=losses)
parser.add_argument('--lr',         type=float, default=0.5  )
parser.add_argument('--beta',       type=float, default=0.9  )
parser.add_argument('--wd',         type=float, default=0.01 )

args = parser.parse_args()

torch.manual_seed(args.seed)
numpy.random.seed(args.seed)


def loss_function(output, target):

    if args.loss == 'mse':
        onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
        error = (output - onehot * math.sqrt(output.shape[1])).square().mean(dim=1)

    elif args.loss == 'xent':
        error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)

    return error.mean()


if __name__ == '__main__':

    _getBatch, input_dim, output_dim = getIterator(dataset="cifar10", batch_size=args.batch_size)

    if args.arch == "resmlp":
        net = ResMLP(   width = args.width,
                        num_blocks = args.depth,
                        block_depth = args.blockdepth,
                        input_dim = numpy.prod(input_dim),
                        output_dim = output_dim
                    )
        def getBatch(train):
            data, target = _getBatch(train)
            return data.flatten(start_dim=1), target

    if not args.cpu: net = net.cuda()

    net.initialize()

    results = {"train_loss":[], "test_loss":[], "train_acc":[], "test_acc":[]}
    os.makedirs(args.logdir, exist_ok=True)
    pickle.dump(vars(args), open( os.path.join(args.log_dir, 'args.pickle'), "wb" ) )

    for step in (pbar := trange(args.train_steps)):

        data, target = getBatch(train = True)

        if not args.cpu: data, target = data.cuda(), target.cuda()

        output = net(data)
        loss = loss_function(output, target)
        loss.backward()

        net.update(args.lr * (1 - step / args.train_steps), beta=args.beta, wd=args.wd)
        net.zero_grad()

        acc = (output.argmax(dim=1) == target).sum() / target.numel()
        pbar.set_description(f"acc: {acc.item():.4f}")

        results["train_loss"].append(loss.item())
        results["train_acc"].append(acc.item())

        if step % args.log_interval == 0:
            loss = 0
            acc = 0
            for _ in range(args.test_steps):
                data, target = getBatch(train = False)
                if not args.cpu: data, target = data.cuda(), target.cuda()

                output = net(data)
                loss += loss_function(output, target)
                acc += (output.argmax(dim=1) == target).sum() / target.numel()

            results["test_loss"].append(loss.item() / args.test_steps)
            results["test_acc"].append(acc.item() / args.test_steps)

            pickle.dump(results, open( os.path.join(args.log_dir, 'results.pickle'), "wb" ) )
            torch.save(net.state_dict(), os.path.join(args.log_dir, 'net.checkpoint'))

            if step > 0 and math.isnan(loss): break
