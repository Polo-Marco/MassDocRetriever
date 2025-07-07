# rerankers/bert_doc_reranker/trainer.py

import torch
from accelerate import Accelerator
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class BertDocTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=16,
        lr=2e-5,
        num_epochs=3,
        save_dir="bert_doc_reranker_ckpt",
        save_metric="f1",
        val_steps=200,  # <--- new: evaluate every val_steps
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        assert save_metric in (
            "f1",
            "recall",
            "precision",
        ), "save_metric must be 'f1', 'recall', or 'precision'"
        self.save_metric = save_metric
        self.val_steps = val_steps

    def compute_metrics(self, model, dataloader, accelerator):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
                labels = batch["labels"].detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"precision": precision, "recall": recall, "f1": f1}

    def train(self):
        accelerator = Accelerator()
        device = accelerator.device
        model = self.model.to(device)

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
        if self.val_dataset:
            val_loader = accelerator.prepare(val_loader)

        best_metric = 0.0
        global_step = 0

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                global_step += 1

                # --- Evaluate every val_steps ---
                if self.val_dataset and (global_step % self.val_steps == 0):
                    metrics = self.compute_metrics(model, val_loader, accelerator)
                    print(
                        f"[Step {global_step}] Validation - Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
                    )
                    metric_value = metrics[self.save_metric]
                    if metric_value > best_metric:
                        best_metric = metric_value
                        accelerator.unwrap_model(model).save_pretrained(self.save_dir)
                        print(
                            f"New best model saved at {self.save_dir} ({self.save_metric}={best_metric:.4f})"
                        )
            print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

            # --- Also evaluate at epoch end ---
            if self.val_dataset:
                metrics = self.compute_metrics(model, val_loader, accelerator)
                print(
                    f"Epoch End Validation - Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
                )
                metric_value = metrics[self.save_metric]
                if metric_value > best_metric:
                    best_metric = metric_value
                    accelerator.unwrap_model(model).save_pretrained(self.save_dir)
                    print(
                        f"New best model saved at {self.save_dir} ({self.save_metric}={best_metric:.4f})"
                    )

        accelerator.wait_for_everyone()
        print("Training done.")
