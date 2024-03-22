import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class Model:
    def __init__(self, transformer_path:str, save_model_path:str, num_labels:int, learning_rate:float, weight_decay:float, device:torch.device):
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.save_model_path = save_model_path
        self.tokenizer = DebertaTokenizer.from_pretrained(transformer_path)
        self.model = DebertaForSequenceClassification.from_pretrained(transformer_path, num_labels=self.num_labels)
        self.model.to(self.device)
        
        # self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.all_train_losses = []
        self.all_val_losses = []

        self.all_train_f1s = []
        self.all_val_f1s = []
        self.all_train_accuracies = []
        self.all_val_accuracies = []
    
    def _train(self, train_loader:DataLoader, epoch:int):
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.model.train()
        
        train_loss = 0.0
        train_preds = []
        train_true = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                
                inputs = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
                
                label = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(**inputs, labels=label)
                
                logits = output.logits.cpu()
                preds = torch.argmax(logits, dim=1).tolist()
                train_preds.extend(preds)
                train_true.extend(label.tolist())
        
                loss = output.loss
                loss.backward()
                optimizer.step()
                
                train_loss += float(loss.detach().cpu().item())
                
                if batch_idx % 10 == 0:
                    tepoch.set_description(f"Epoch {epoch+1}")
                    tepoch.set_postfix(loss=loss.detach().cpu().item() / len(batch))

        train_loss /= len(train_loader)
        
        return train_loss, train_preds, train_true
    
    
    def _eval(self, val_loader:DataLoader, epoch:int):
        self.model.eval()
        
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                
                inputs = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
                
                label = batch["label"].to(self.device)
            
                output = self.model(**inputs, labels=label)
                logits = output.logits.cpu()
                preds = torch.argmax(logits, dim=1).tolist()
                val_preds.extend(preds)
                val_true.extend(label.tolist())
                loss = output.loss
                val_loss += float(loss.detach().cpu().item())
    
        val_loss /= len(val_loader)
        
        return val_loss, val_preds, val_true
    
    
    def training_loop(self, training_loader:DataLoader, val_loader:DataLoader, num_epochs:int):
        for epoch in range(num_epochs):
            train_loss, train_preds, train_true = self._train(training_loader, epoch)
            val_loss, val_preds, val_true = self._eval(val_loader, epoch)
            
            train_accuracy = accuracy_score(train_true, train_preds)
            train_f1 = f1_score(train_true, train_preds)
            
            val_accuracy = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds)
            
            self.all_train_losses.append(train_loss)
            self.all_val_losses.append(val_loss)
            self.all_train_f1s.append(train_f1)
            self.all_val_f1s.append(val_f1)
            self.all_train_accuracies.append(train_accuracy)
            self.all_val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Train F1: {train_f1} | Val F1: {val_f1} | Train Acc: {train_accuracy} | Val Acc: {val_accuracy}")
            
            # if val_f1 >= max(self.all_val_f1s):
            #     torch.save(self.model, self.save_model_path + ".pt")
        
            torch.cuda.empty_cache()
    
    def test(self, test_loader:DataLoader):
        self.model.eval()
        
        test_preds = []
        test_true = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                
                inputs = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

                label = batch["label"].cpu()

                output = self.model(**inputs)
                logits = output.logits.cpu()
                preds = torch.argmax(logits, dim=1).tolist()
            
                test_preds.extend(preds)
                test_true.extend(label.cpu().tolist())
            
            torch.cuda.empty_cache()
    
    def save_metrics(self):
        metrics = {
            "train_losses": self.all_train_losses,
            "val_losses": self.all_val_losses,
            "train_f1s": self.all_train_f1s,
            "val_f1s": self.all_val_f1s,
            "train_accuracies": self.all_train_accuracies,
            "val_accuracies": self.all_val_accuracies,
            "test_preds": self.test_preds,
        }
        
        torch.save(metrics, self.save_model_path + "_metrics.pt")
                
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        
        ax1.plot(self.all_train_losses, label="Train Loss")
        ax1.plot(self.all_val_losses, label="Val Loss")
        ax1.legend()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curve")
        ax1.grid()
        
        ax2.plot(self.all_train_f1s, label="Train F1")
        ax2.plot(self.all_val_f1s, label="Val F1")
        ax2.plot(self.all_train_accuracies, label="Train Acc")
        ax2.plot(self.all_val_accuracies, label="Val Acc")
        ax2.legend()
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1/Accuracy")
        ax2.title("Evalutaion Metrics")
        ax2.grid()
        
        fig.subplots_adjust(hspace=0.5, wspace=0.25)

        loss_curve_path = self.model_path + '_LossCurve.png'
        fig.savefig(loss_curve_path)
    
    
    
    
if __name__ == "__main__":
    pass

