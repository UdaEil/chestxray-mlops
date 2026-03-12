import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ── Config ──────────────────────────────────────────
DATA_DIR   = "data/raw/archive/chest_xray"
EPOCHS     = 5         # higher max — early stopping will kick in
BATCH_SIZE = 32
LR         = 0.001
PATIENCE   = 3           # stop if val_acc doesn't improve for 3 epochs
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── Data ─────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # data augmentation
    transforms.RandomRotation(10),           # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform_train)
val_data   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=transform_val)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)

print(f"Classes: {train_data.classes}")
print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

# ── Model (ResNet18 pretrained + Dropout) ─────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Dropout(0.3),                              # prevents memorization
    nn.Linear(model.fc.in_features, 2)            # 2 classes: NORMAL / PNEUMONIA
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# LR Scheduler — halves LR if val_acc doesn't improve
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=1, factor=0.5, verbose=True
)

# ── MLflow Tracking ───────────────────────────────────
mlflow.set_experiment("chest-xray-pneumonia-v2")

with mlflow.start_run():

    # Log hyperparameters
    mlflow.log_param("epochs",     EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("lr",         LR)
    mlflow.log_param("patience",   PATIENCE)
    mlflow.log_param("model",      "resnet18 + dropout")
    mlflow.log_param("augmentation", "flip + rotation")

    best_val_acc     = 0
    patience_counter = 0

    for epoch in range(EPOCHS):

        # ── Training ──────────────────────────────────
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / total

        # ── Validation ────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        # Step the scheduler
        scheduler.step(val_acc)

        # ── Log metrics to MLflow ──────────────────────
        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_acc",  train_acc,                      step=epoch)
        mlflow.log_metric("val_loss",   val_loss / len(val_loader),     step=epoch)
        mlflow.log_metric("val_acc",    val_acc,                        step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ── Early Stopping + Save Best Model ──────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)
            print(f"  ✅ New best model saved! Val Acc: {best_val_acc:.3f}")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best Val Acc: {best_val_acc:.3f}")
                break

    # ── Log best model to MLflow ──────────────────────
    model.load_state_dict(torch.load("models/best_model.pt"))
    mlflow.pytorch.log_model(model, name="best_model")
    mlflow.log_param("best_val_acc", best_val_acc)

    print(f"\n✅ Training complete!")
    print(f"   Best Val Acc : {best_val_acc:.3f}")
    print(f"   Model saved  : models/best_model.pt")
    print(f"   MLflow run   : check http://localhost:5000")