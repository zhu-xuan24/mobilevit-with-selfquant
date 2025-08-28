"""
  資料前處理+推論示例
"""

import torch
from model import mobile_vit_xx_small
from PIL import Image
from torchvision import transforms

# 建立模型
model = mobile_vit_xx_small(pretrained=True)  # 如果有預訓練權重
model.eval()                                  # 推論模式

# 資料前處理
eval_transform = transforms.Compose([
    transforms.Resize(256),               # 縮放最短邊到 256，並保持長寬比
    transforms.CenterCrop(224),           # 從中間裁切 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 載入圖片&前處理
img = Image.open("cat.jpg").convert("RGB")
x = eval_transform(img).unsqueeze(0)      # 前處理，然後reshape to [1, 3, 224, 224]

# 推論
with torch.no_grad():
    out = model(x)
    pred = torch.argmax(out, dim=1)

# 讀取標籤檔
with open("imagenet1k_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# 轉換成類別名稱
pred_idx = pred.item()
pred_label = labels[pred_idx]
print(f"預測類別: {pred_label}")
