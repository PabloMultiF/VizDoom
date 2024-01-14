import argparse
from train import train
from test import test

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
        
    args.add_argument('-lr', type=float, default=0.0001)
    args.add_argument('-gamma', type=float, default=0.7)
    args.add_argument('-lam', type=float, default=0.95)
    args.add_argument('-eps', type=float, default=0.1)
    args.add_argument('-c1', type=float, default=1.0)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-clip', type=float, default=0.1)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-batch_size', type=int, default=129)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-cicles', type=int, default=10000)
    args.add_argument('-train', default='True', choices=('True','False'))

    arguments = args.parse_args()

    if(arguments.train == 'True'):
        print('Modo: Entrenamiento')
        train(arguments)
    else:
        print('Modo: Testeo')
        test(arguments)