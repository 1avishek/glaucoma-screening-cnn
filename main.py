from args import get_args
from trainer import Trainer

def main():
    args = get_args()
    print("Using device:", args.device)

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
