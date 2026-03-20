import torch.optim as optim
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

class BertClassifierTrainer():
    def __init__(self, train_loader, val_loader, test_loader, classes_labels):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes_labels = classes_labels
        
    def train(self, classifier, config):
        self.classifier = classifier
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.classifier.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.classifier.to(self.device)
        best_val_loss = np.inf

        ckpt_filename = f"{config['export_folder']}{config['exp_name']}"
        
        total_steps = len(self.train_loader) * config['epochs']
        warmup_steps = int(total_steps * 0.1) 
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

        for epoch in range(config['epochs']):
            self.classifier.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} - Training"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                logits = self.classifier(batch)

                loss = loss_function(logits, batch['labels'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

            avg_loss = total_loss / len(self.train_loader)
            wandb.log({"avg_train_loss": avg_loss})

            self.classifier.eval() 
            val_loss = 0
            with torch.no_grad(): 
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} - Validation"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch['labels']
                    logits = self.classifier(batch)

                    loss = loss_function(logits, labels)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(self.val_loader)
                wandb.log({"avg_val_loss": avg_val_loss})
                if avg_val_loss < best_val_loss:
                    print(f"Saving best checkpoint at epoch {epoch+1}")
                    self.save_model(f"{ckpt_filename}_best.pt")
                    best_val_loss = avg_val_loss
        
        self.save_model(f"{ckpt_filename}_last.pt")
        self.log_artifact(f"{config['exp_name']}_last", f"{ckpt_filename}_last.pt")

        self.log_artifact(f"{config['exp_name']}_best", f"{ckpt_filename}_best.pt")

        report = self.run_evaluation(mode='val', filename=f"{ckpt_filename}_best.pt")
        return report

    def run_evaluation(self, mode, filename=None):
        loader = self.test_loader if mode == 'test' else self.val_loader

        if filename != None:
            self.classifier = torch.load(filename, weights_only=False)

        self.classifier.eval()
        self.classifier.to(self.device)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Prediction"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch['labels']
                logits = self.classifier(batch)
                predicted = torch.argmax(logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        report = classification_report(y_true, y_pred, output_dict=True, target_names=self.classes_labels)
        wandb.log({f"{mode}_macro_avg_metrics": report['macro avg']}) 

        wandb.log({f"{mode}_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=self.classes_labels
        )})

        return report    
    
    def test(self, filename):
        return self.run_evaluation('test', filename)

    def save_model(self, filename):
        torch.save(self.classifier, filename)

    def log_artifact(self, artifact_name, filename):
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(filename)
        wandb.log_artifact(artifact)
