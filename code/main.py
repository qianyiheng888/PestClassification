import os
import torch
import timm.scheduler
import numpy as np
from Dataset import loaders
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timm
from PestModel import PestModel
from fix_random_seed import setup_seed
torch.cuda.set_device(0)
os.environ['TORCH_HOME'] = '/hy-tmp/models'

base_lr = 4e-4
BATCH_SIZE = 32
NUM_EPOCHES = 150
LEARN_RATE = base_lr * BATCH_SIZE / 256
setup_seed(3407)

#model = timm.create_model('resnet50', pretrained=True, num_classes=102)

model = PestModel(num_classes=102,dim = 1536)

model = model.cuda()
train_dataloader, test_dataloader, val_dataloader = loaders(BATCH_SIZE)
LossFunc = torch.nn.CrossEntropyLoss()
num_training_steps = NUM_EPOCHES * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=LEARN_RATE)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

f = open('./result/result.txt', 'w', encoding='utf-8')
for epoch in range(NUM_EPOCHES):
    #训练loop
    model.train()
    epoch_train_loss=0.0
    pbar = tqdm(train_dataloader)
    for train_batch, (train_image, train_label) in enumerate(pbar):
        train_image,train_label = train_image.cuda(),train_label.cuda()  # 将数据送到GPU
        #with torch.cuda.amp.autocast():
        train_predictions = model(train_image)
        batch_train_loss = LossFunc(train_predictions,train_label)
        optimizer.zero_grad()  # 梯度清零
        batch_train_loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch=epoch)
        epoch_train_loss += batch_train_loss.item()
        pbar.set_description(f'epoch:{epoch} train_loss:{batch_train_loss:2.4f}')
        
    contents = f'train_loss:{epoch_train_loss/np.ceil(len(train_dataloader.dataset)/BATCH_SIZE):2.4f}'
    print(contents)
    f.write(f'epoch:{epoch}, ' + contents)
    #验证loop
    model.eval()
    epoch_test_loss=0.0
    epoch_test_acc=0.0
    pbar = tqdm(val_dataloader)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_batch, (test_image, test_label) in enumerate(pbar):
            test_image, test_label = test_image.cuda(), test_label.cuda()
            
            predictions = model(test_image)
            test_loss = LossFunc(predictions,test_label)
            epoch_test_loss+=test_loss.item()
            pred_label = predictions.argmax(dim=1)
            y_true.extend(test_label.cpu().numpy().tolist())
            y_pred.extend(pred_label.cpu().numpy().tolist())
            pbar.set_description(f'epoch:{epoch}(val) val_loss:{test_loss:2.4f}')
    acc = accuracy_score(y_pred=y_pred, y_true=y_true)
    pre = precision_score(y_pred=y_pred, y_true=y_true, average='weighted')
    rec = recall_score(y_pred=y_pred, y_true=y_true, average='weighted')
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
    contents = f'val_loss:{epoch_test_loss/np.ceil(len(val_dataloader.dataset)/BATCH_SIZE):2.4f} val_acc:{acc:1.4f} f1 score:{f1:1.4f}'
    print(contents)
    f.write(', '+contents+'\n')
    if epoch == NUM_EPOCHES-1:#测试loop
        epoch_test_loss=0.0
        epoch_test_acc=0.0
        pbar = tqdm(test_dataloader)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for test_batch, (test_image, test_label) in enumerate(pbar):
                test_image, test_label = test_image.cuda(), test_label.cuda()
                
                predictions = model(test_image)
                test_loss = LossFunc(predictions,test_label)
                epoch_test_loss+=test_loss.item()
                pred_label = predictions.argmax(dim=1)
                y_true.extend(test_label.cpu().numpy().tolist())
                y_pred.extend(pred_label.cpu().numpy().tolist())
                pbar.set_description(f'epoch:{epoch}(test) test_loss:{test_loss:2.4f}')
        acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        pre = precision_score(y_pred=y_pred, y_true=y_true, average='weighted')
        rec = recall_score(y_pred=y_pred, y_true=y_true, average='weighted')
        f1 = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
        contents = f'test_loss:{epoch_test_loss/np.ceil(len(test_dataloader.dataset)/BATCH_SIZE):2.4f} accuracy:{acc:1.4f} f1 score:{f1:1.4f}'
        print(contents)
        f.write(', '+contents+'\n')
f.close()
