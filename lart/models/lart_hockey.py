import os
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightning import LightningModule
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmetrics import MeanMetric

from lart.models.components.lart_transformer.transformer import lart_transformer
from lart.utils import get_pylogger

log = get_pylogger(__name__)


class HockeyLART_LitModule(LightningModule):
    """LART's 3D-pose transformer, trained from scratch as a single-label
    per-clip classifier on the 11-class hockey action dataset.

    Unlike LART_LitModule (AVA/Kinetics), there's no SMPL-parameter or
    3D-location supervision here (our pose/joint tensors are MHR-derived
    inputs, not SMPL targets) and no appearance stream, so this reuses only
    the lart_transformer backbone + its action_head_kinetics classification
    head (repurposed for our 11 classes via configs.kinetics.num_action_classes),
    with a plain class-weighted cross-entropy loss and a mean-class-accuracy /
    confusion-matrix validation protocol (matching degcn-skating-actions'
    analyze_results.py, for direct comparability with that baseline).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = self.hparams.cfg

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.encoder = lart_transformer(
            opt=self.cfg,
            depth=self.cfg.transformer.depth,
            heads=self.cfg.transformer.heads,
            mlp_dim=self.cfg.transformer.mlp_dim,
            dim_head=self.cfg.transformer.dim_head,
            dropout=self.cfg.transformer.dropout,
            emb_dropout=self.cfg.transformer.emb_dropout,
            droppath=self.cfg.transformer.droppath,
            device=self.device,
        )

        self.num_classes = self.cfg.kinetics.num_action_classes
        if self.cfg.get("use_class_weighted_loss", True):
            self.register_buffer("class_weights", self._compute_class_weights())
        else:
            self.register_buffer("class_weights", torch.ones(self.num_classes, dtype=torch.float32))

        # NOTE: lart_transformer's own action_head_kinetics operates on
        # `class_token`, which is concatenated onto the sequence *after* the
        # transformer runs (see lart_transformer.forward) rather than being
        # attended to as an input token -- it's a constant learned parameter,
        # identical for every sample, independent of the input. Verified this
        # empirically (identical logits across a whole batch) before adding
        # this classifier: we instead mean-pool the transformer's actual
        # per-frame outputs (`pose_tokens`) over the valid (non-padded)
        # frames and classify that.
        self.classifier = torch.nn.Linear(self.cfg.in_feat, self.num_classes)

        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        self._val_preds = []
        self._val_gts = []

    def _compute_class_weights(self):
        train_path = os.path.join(self.cfg.data_dir, "hockey", "train.pkl")
        with open(train_path, "rb") as f:
            clips = pickle.load(f)
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for c in clips:
            counts[c["label"]] += 1
        weights = counts.sum() / (self.num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def forward(self, tokens, mask_type):
        output, vq_loss = self.encoder(tokens, mask_type)
        # output = [class_token (constant, unused), pose_tokens (BS, T*P, dim)]
        pose_tokens = output[:, self.cfg.max_people :, :]
        BS = pose_tokens.shape[0]
        pose_tokens = pose_tokens.view(BS, self.cfg.frame_length, self.cfg.max_people, self.cfg.in_feat)

        has_detection = tokens["has_detection"][:, :, 0, :]  # ego person, (BS, T, 1)
        valid_counts = has_detection.sum(dim=1).clamp(min=1.0)  # (BS, 1)
        pooled = (pose_tokens[:, :, 0, :] * has_detection).sum(dim=1) / valid_counts  # (BS, in_feat)

        logits = self.classifier(pooled)
        return logits, vq_loss

    def step(self, batch: Any, mask_type: str):
        input_data, labels = batch
        logits, vq_loss = self.forward(input_data, mask_type)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.dtype))
        return loss, logits, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.step(batch, self.cfg.mask_type)
        self.train_loss(loss.item())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_start(self):
        self._val_preds = []
        self._val_gts = []

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, labels = self.step(batch, self.cfg.mask_type_test)
        self.val_loss(loss.item())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = torch.argmax(logits, dim=-1)
        self._val_preds.append(preds.detach().cpu())
        self._val_gts.append(labels.detach().cpu())
        return {"loss": loss}

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        preds = torch.cat(self._val_preds).numpy()
        gts = torch.cat(self._val_gts).numpy()

        mean_class_acc = balanced_accuracy_score(gts, preds)
        overall_acc = float((preds == gts).mean())
        cm = confusion_matrix(gts, preds, labels=list(range(self.num_classes)))

        self.log("val/mean_class_acc", mean_class_acc, prog_bar=True)
        self.log("val/overall_acc", overall_acc, prog_bar=True)
        log.info(f"val mean_class_acc={mean_class_acc:.4f} overall_acc={overall_acc:.4f}")
        log.info(f"confusion matrix:\n{cm}")

        out_path = os.path.join(self.cfg.storage_folder, "results", f"confmat_epoch_{self.current_epoch}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({"confusion_matrix": cm, "mean_class_acc": mean_class_acc, "overall_acc": overall_acc}, f)

    def test_step(self, batch: Any, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_start(self):
        self.on_validation_start()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg.solver.lr,
            weight_decay=self.cfg.solver.weight_decay,
            betas=(0.9, 0.95),
        )

        warmup_epochs = self.cfg.solver.warmup_epochs

        def warm_start_and_cosine_annealing(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (
                1.0
                + np.cos(np.pi * ((epoch + 1) - warmup_epochs) / (self.trainer.max_epochs - warmup_epochs))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_start_and_cosine_annealing)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
