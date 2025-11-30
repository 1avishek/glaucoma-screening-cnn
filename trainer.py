import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from args import get_args
from dataset import FundusSegmentationDataset, get_refuge_file_lists, get_k_fold_splits
from model import build_model
from utils import set_seed, dice_score


class Trainer:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        proj_root = Path(args.project_root)
        self.output_dir = proj_root / args.output_dir / f"fold_{args.fold_index}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.txt"
        self.log_path.write_text("")

        # Load file lists
        imgs, masks = get_refuge_file_lists(args.project_root)
        tr_i, tr_m, va_i, va_m = get_k_fold_splits(
            imgs, masks, args.num_folds, args.fold_index, args.seed
        )

        # Transforms
        train_tf = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(p=0.3),
            A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
            ToTensorV2(),
        ])

        val_tf = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
            ToTensorV2(),
        ])

        # Datasets
        self.train_ds = FundusSegmentationDataset(tr_i, tr_m, train_tf)
        self.val_ds = FundusSegmentationDataset(va_i, va_m, val_tf)

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        self.model = build_model(args.backbone, args.num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.best_dice = 0.0

    def log(self, msg: str):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def compute_loss(self, logits, masks):
        ce = F.cross_entropy(logits, masks)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        dices = dice_score(preds, masks, self.args.num_classes)
        mean_dice = torch.stack([d for d in dices.values()]).mean()
        dice_loss = 1 - mean_dice

        return ce + dice_loss, dices

    def train(self):
        for epoch in range(1, self.args.num_epochs + 1):
            self.log("")
            self.log(f"Epoch {epoch}/{self.args.num_epochs}")

            tr_loss = self.train_one(epoch)
            va_loss, va_dice = self.validate(epoch)

            mean_va = torch.stack([torch.tensor(v) for v in va_dice.values()]).mean().item()
            self.log(f"Val dice={mean_va:.4f}")

            if mean_va > self.best_dice:
                self.best_dice = mean_va
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
                self.log("🔥 New Best Model Saved")

    def train_one(self, epoch):
        self.model.train()
        total = 0.0

        for img, mask, _ in tqdm(self.train_loader, desc=f"Train {epoch}"):
            img, mask = img.to(self.device), mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(img)
            loss, _ = self.compute_loss(logits, mask)
            loss.backward()
            self.optimizer.step()

            total += loss.item() * img.size(0)

        return total / len(self.train_loader.dataset)

    def validate(self, epoch):
        self.model.eval()
        total = 0.0
        dice_sum = {1: 0.0, 2: 0.0}
        count = 0

        with torch.no_grad():
            for img, mask, _ in tqdm(self.val_loader, desc=f"Val {epoch}"):
                img, mask = img.to(self.device), mask.to(self.device)

                logits = self.model(img)
                loss, dices = self.compute_loss(logits, mask)

                total += loss.item() * img.size(0)
                for c in dice_sum:
                    dice_sum[c] += dices[c].item() * img.size(0)

                count += img.size(0)

        avg_dice = {c: dice_sum[c] / count for c in dice_sum}
        return total / len(self.val_loader.dataset), avg_dice
