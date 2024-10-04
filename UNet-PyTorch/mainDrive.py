import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm

from unet import UNet
from drive_dataset import DriveDataset
from metrics import f1_score,iou,recall,dice_coef,accuracy,precision,dice_loss, iou_loss_smooth
import matplotlib.pyplot as plt

if __name__ == "__main__":
    LEARNING_RATE = 3e-6  # Ajuste da taxa de aprendizado
    BATCH_SIZE = 32  # Ajuste do tamanho do lote
    EPOCHS = 100  # Ajuste do número de épocas
    DATA_PATH = "/home/os/Documentos/mestrado usp/materias/MAC5921-Deep Learning/EP2/UNet-PyTorch/data_drive/traning"
    DATA_PATH_TESTE = "/home/os/Documentos/mestrado usp/materias/MAC5921-Deep Learning/EP2/UNet-PyTorch/data_drive/testing"
    MODEL_SAVE_PATH = "./models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DriveDataset(DATA_PATH, resize_size=(128, 128), transform=train_transforms)
    test_dataset = DriveDataset(DATA_PATH_TESTE, resize_size=(128, 128))


    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = iou_loss_smooth

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(EPOCHS), colour="blue", desc="Epoch", leave=True):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader, colour="green",
                                            desc="Training", leave=False)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_losses.append(train_loss)

        model.eval()

        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(val_dataloader):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        val_loss = val_running_loss / (idx + 1)
        val_losses.append(val_loss)

        torch.cuda.empty_cache()

    # Salvar o modelo
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # Calcula a perda total no final do treinamento
    total_train_loss = sum(train_losses) / len(train_losses)
    total_val_loss = sum(val_losses) / len(val_losses)
    total_loss = total_train_loss + total_val_loss

    # Imprime as perdas totais
    print("-" * 30)
    print(f"Total Train Loss: {total_train_loss:.4f}")
    print(f"Total Validation Loss: {total_val_loss:.4f}")
    print(f"Total Loss: {total_loss:.4f}")
    print("-" * 30)

    # Plotando a convergência da função de perda
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('imagens/grafico_convergencia.png') # Salva o gráfico como imagem

    # Carregar o modelo treinado
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))

    # Variáveis para armazenar métricas
    test_running_loss = 0
    test_running_dice = 0
    test_running_iou = 0
    test_running_accuracy = 0
    test_running_precision = 0
    test_running_recall = 0
    test_running_f1 = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(test_dataloader, colour="yellow", desc="Testando", leave=False)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            # Aplica threshold para obter a segmentação binária
            y_pred = torch.sigmoid(model(img))
            y_pred_bin = (y_pred > 0.5).float()

            loss = criterion(y_pred, mask)

            test_running_loss += loss.item()
            test_running_dice += dice_coef(mask, y_pred_bin).item()
            test_running_iou += iou(mask, y_pred_bin).item()
            test_running_accuracy += accuracy(mask, y_pred_bin).item()
            test_running_precision += precision(mask, y_pred_bin).item()
            test_running_recall += recall(mask, y_pred_bin).item()
            test_running_f1 += f1_score(mask, y_pred_bin).item()

        test_loss = test_running_loss / (idx + 1)
        test_dice = test_running_dice / (idx + 1)
        test_iou = test_running_iou / (idx + 1)
        test_accuracy = test_running_accuracy / (idx + 1)
        test_precision = test_running_precision / (idx + 1)
        test_recall = test_running_recall / (idx + 1)
        test_f1 = test_running_f1 / (idx + 1)

    print("-" * 30)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("-" * 30)
