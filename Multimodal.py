import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import chardet
from torchvision.models import ResNet18_Weights
from tqdm import tqdm 

class MultimodalDataset(Dataset):
    """
    自定义多模态数据集，处理文本和图像数据。

    Args:
        df (pd.DataFrame): 包含数据的DataFrame。
        tokenizer (BertTokenizer): 用于文本编码的BERT分词器。
        transform (torchvision.transforms.Compose, optional): 图像预处理方法。
        is_test (bool, optional): 是否为测试集，用于决定是否返回标签。
    """
    def __init__(self, df, tokenizer, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项。

        Args:
            idx (int): 数据项的索引。

        Returns:
            dict: 包含guid, input_ids, attention_mask, image, (label)的字典。
        """
        guid = self.df.loc[idx, 'guid']
        if not self.is_test:
            label = self.df.loc[idx, 'label']
        else:
            label = -1

        text_path = os.path.join(DATA_DIR, f'{guid}.txt')
        with open(text_path, 'rb') as f:
            raw_data = f.read(10000)  
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(text_path, 'r', encoding=encoding, errors='ignore') as f:
            text = f.read()

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()  
        attention_mask = encoding['attention_mask'].squeeze() 

        image_path_jpg = os.path.join(DATA_DIR, f'{guid}.jpg')
        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        else:
            raise FileNotFoundError(f'Image file for GUID {guid} not found as .jpg.')

        image = Image.open(image_path).convert('RGB')  
        if self.transform:
            image = self.transform(image) 

        item = {
            'guid': guid,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image
        }

        if not self.is_test:
            item['label'] = torch.tensor(label, dtype=torch.long)

        return item

class CrossAttentionFusion(nn.Module):
    """
    改进版的交叉注意力融合层，用于融合文本和图像特征。

    Args:
        dim (int, optional): 特征维度。默认值为512。
        num_heads (int, optional): 注意力头数。默认值为8。
        dropout (float, optional): Dropout比例。默认值为0.1。
    """
    def __init__(self, dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)  
        self.kv_proj = nn.Linear(dim, 2 * dim)  

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, text_feat, image_feat):
        """
        前向传播，融合文本和图像特征。

        Args:
            text_feat (torch.Tensor): 文本特征，形状为(batch_size, dim)。
            image_feat (torch.Tensor): 图像特征，形状为(batch_size, dim)。

        Returns:
            torch.Tensor: 融合后的特征，形状为(batch_size, dim)。
        """
        residual = text_feat

        text_feat = self.norm1(text_feat)
        image_feat = self.norm1(image_feat)

        q = self.q_proj(text_feat)  
        kv = self.kv_proj(image_feat)  
        k, v = kv.chunk(2, dim=-1)  

        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        attn_weights = F.softmax(attn_weights, dim=-1)  
        attn_weights = self.attn_dropout(attn_weights)  

        attn_output = torch.matmul(attn_weights, v) 
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)  

        attn_output = residual + attn_output.squeeze(1) 

        ffn_output = self.ffn(self.norm2(attn_output))
        fused_feature = attn_output + ffn_output

        return fused_feature

class Classifier(nn.Module):
    """
    多模态情感分类模型，结合文本和图像特征进行分类。

    Args:
        text_model_name (str, optional): 文本编码器的预训练模型名称。默认值为'bert-base-uncased'。
        num_classes (int, optional): 分类的类别数。默认值为3。
    """
    def __init__(self, text_model_name='bert-base-uncased', num_classes=3):
        super().__init__()
        
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.text_model.config.hidden_size 

        self.image_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for name, param in self.image_model.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.image_hidden_size = self.image_model.fc.in_features  
        self.image_model.fc = nn.Identity()

        self.text_proj = nn.Linear(768, 512)  
        self.image_proj = nn.Linear(512, 512) 

        self.cross_attn = CrossAttentionFusion(dim=512, num_heads=8)


        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        """
        前向传播，处理文本和图像输入并进行分类。

        Args:
            input_ids (torch.Tensor): 文本的输入ID，形状为(batch_size, MAX_SEQ_LENGTH)。
            attention_mask (torch.Tensor): 文本的注意力掩码，形状为(batch_size, MAX_SEQ_LENGTH)。
            image (torch.Tensor): 图像数据，形状为(batch_size, 3, 224, 224)。

        Returns:
            torch.Tensor: 分类的logits，形状为(batch_size, num_classes)。
        """
        if EXP_MODE == 'text':
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = self.text_proj(text_outputs.pooler_output)
            
            image_features = torch.zeros(text_features.size(0), 512).to(device)

        elif EXP_MODE == 'image':
            image_features = self.image_proj(self.image_model(image))

            text_features = torch.zeros(image_features.size(0), 512).to(device)

        else:
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = self.text_proj(text_outputs.pooler_output) 

            image_features = self.image_proj(self.image_model(image))  

        fused_features = self.cross_attn(text_features, image_features)  # (bs, 512)

        logits = self.classifier(fused_features)
        return logits

def train_epoch(model, dataloader, criterion, optimizer, window_size=8):
    """
    训练模型一个epoch。

    Args:
        model (nn.Module): 训练的模型。
        dataloader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        window_size (int, optional): 窗口大小，用于计算平均损失。默认值为8。

    Returns:
        float: 该epoch的平均训练损失。
    """
    model.train()  
    total_loss = 0
    batch_count = 0
    window_loss = 0

    for idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad() 
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)

        window_loss += loss.item()
        batch_count += 1

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        if batch_count >= window_size:
            total_loss += window_loss / window_size
            window_loss = 0
            batch_count = 0

    if batch_count > 0:
        total_loss += window_loss / batch_count

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, criterion, window_size=8):
    """
    在验证集上评估模型。

    Args:
        model (nn.Module): 评估的模型。
        dataloader (DataLoader): 验证数据加载器。
        criterion (nn.Module): 损失函数。
        window_size (int, optional): 窗口大小，用于计算平均损失。默认值为8。

    Returns:
        tuple: (平均验证损失, 验证准确率)。
    """
    model.eval()
    total_loss = 0
    batch_count = 0
    window_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, images) 
            loss = criterion(outputs, labels) 

            window_loss += loss.item()
            batch_count += 1
            if batch_count >= window_size:
                total_loss += window_loss / window_size 
                window_loss = 0 
                batch_count = 0 

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if batch_count > 0:
        total_loss += window_loss / batch_count

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def predict_test_set(model, test_loader, device, original_test_df):
    """
    在测试集上进行预测，并生成包含预测标签的DataFrame。

    Args:
        model (nn.Module): 训练好的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 使用的设备（CPU或GPU）。
        original_test_df (pd.DataFrame): 原始测试集DataFrame，用于保持GUID顺序。

    Returns:
        pd.DataFrame: 包含预测标签的DataFrame。
    """
    model.eval()
    predictions = []
    guids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            guids_batch = batch['guid']

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)

            outputs = model(input_ids, attention_mask, images)
            _, preds = torch.max(outputs, dim=1) 

            predictions.extend(preds.cpu().numpy())
            guids.extend(guids_batch)

    missing_guids = set(original_test_df['guid']) - set(guids)
    if missing_guids:
        raise RuntimeError(f"Missing predictions for {len(missing_guids)} GUIDs, examples: {list(missing_guids)[:5]}")

    result_df = original_test_df.copy()
    result_df['tag'] = [label_mapping_inv[pred] for pred in predictions]

    assert list(result_df['guid']) == list(original_test_df['guid']), "GUID顺序不一致"
    assert not result_df['tag'].isnull().any(), "存在未处理的null标签"

    return result_df

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 实验模式和超参数选择
    EXP_MODE = 'text'
    assert EXP_MODE in ['text', 'image', 'multimodal'], "Invalid experiment mode"
    print(f"\n===== Running {EXP_MODE.upper()} Experiment =====")
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5
    MAX_SEQ_LENGTH = 128

    DATA_DIR = 'data'
    TRAIN_LABELS_PATH = os.path.join(DATA_DIR, 'train.txt')
    TEST_PATH = os.path.join(DATA_DIR, 'test_without_label.txt')

    train_df = pd.read_csv(TRAIN_LABELS_PATH, sep=',')
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2} 
    train_df['label'] = train_df['tag'].map(label_mapping) 

    label_mapping_inv = {v: k for k, v in label_mapping.items()}

    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=453, stratify=train_df['label'])

    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])   
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = MultimodalDataset(train_data, tokenizer, transform=image_transforms)
    val_dataset = MultimodalDataset(val_data, tokenizer, transform=image_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Classifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=5e-4
    )

    best_val_loss = float('inf')
    patience = 8
    no_improve = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        val_loss, val_acc = eval_epoch(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')  
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break 

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    plt.figure(figsize=(10, 5))
    actual_epochs = len(train_losses)
    epochs_range = range(1, actual_epochs + 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    min_val_loss_epoch = np.argmin(val_losses) + 1  
    min_val_loss = val_losses[min_val_loss_epoch - 1]
    plt.scatter(min_val_loss_epoch, min_val_loss, c='red', s=50, zorder=5)
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch}',
                 xy=(min_val_loss_epoch, min_val_loss),
                 xytext=(min_val_loss_epoch + 1, min_val_loss + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(f'{EXP_MODE}_loss_curve.png') 
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    max_val_acc_epoch = np.argmax(val_accuracies) + 1
    max_val_acc = val_accuracies[max_val_acc_epoch - 1]
    plt.scatter(max_val_acc_epoch, max_val_acc, c='green', s=50, zorder=5)
    plt.annotate(f'Max Acc: {max_val_acc:.4f}\nEpoch: {max_val_acc_epoch}',
                 xy=(max_val_acc_epoch, max_val_acc),
                 xytext=(max_val_acc_epoch + 1, max_val_acc - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig(f'{EXP_MODE}_accuracy_curve.png')
    plt.show()

    torch.save(model.state_dict(), 'multimodal_sentiment_model.pth')
    model.load_state_dict(torch.load('best_model.pth'))

    test_df = pd.read_csv(TEST_PATH, sep=',')
    print(f"Test samples: {len(test_df)}")
    test_df['guid'] = test_df['guid'].astype(int)  

    test_dataset = MultimodalDataset(test_df, tokenizer, transform=image_transforms, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=0  
    )

    result_df = predict_test_set(model, test_loader, device, original_test_df=test_df)
    result_df.to_csv('predictions.txt', sep=',', index=False, header=True)
    print("Predictions saved to predictions.txt")
