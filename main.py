import argparse

from training import Trainer
from modeling import KWSModel


def main(args):

    model = KWSModel()

    if args.train:
        trainer = Trainer(model)
        trainer.train()
    elif args.test:
        pass

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train", action="store_true")
    # parser.add_argument("--ckpt")

    # args = parser.parse_args()
    # main(args)
    model = KWSModel()
    trainer = Trainer(model)
    trainer.train()
