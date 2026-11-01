import argparse

from training import Trainer


def main(args):
    # trainer = Trainer(mode="train")
    # trainer.train()
    if args.train: 
        trainer = Trainer(mode="train")
        trainer.train(args.ckpt)
    if args.train_ssl: 
        trainer = Trainer(mode="train")
        trainer.train_ssl(args.ckpt)
    elif args.test_ctc:
        trainer = Trainer(mode="test")
        trainer.test_ctc(args.ckpt)
    elif args.test_ntc:
        trainer = Trainer(mode="test")
        trainer.test_ntc(args.ckpt)
    elif args.test_noisy:
        trainer = Trainer(mode="test")
        trainer.test_ctc_noisy(args.wav_path)
    elif args.test_streaming:
        trainer = Trainer(mode="test")
        trainer.test_streaming(
            args.wav_path,
            args.ckpt,
            args.streaming_threshold,
        )
    elif args.test_compare:
        trainer = Trainer(mode="test")
        trainer.compare_curve()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-ssl", action="store_true")
    parser.add_argument("--test-ctc", action="store_true")
    parser.add_argument("--test-ntc", action="store_true")
    parser.add_argument("--test-compare", action="store_true")
    parser.add_argument("--test-noisy", action="store_true")
    parser.add_argument("--test-streaming", action="store_true")
    parser.add_argument("--wav-path", default=None, help="Path to the WAV file to evaluate")
    parser.add_argument("--streaming-threshold", type=float, default=-6.0)
    parser.add_argument("--ckpt", default=None)

    args = parser.parse_args()
    if (args.test_noisy or args.test_streaming) and not args.wav_path:
        parser.error("--wav-path is required with --test-noisy or --test-streaming")
    main(args)