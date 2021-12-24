import torch
import modelnn
import modeltrn
import train

model = modelnn.model
optimizer = modeltrn.optimizer
loss_fn = modeltrn.loss_fn
train_data_loader = modeltrn.train_data_loader
test_data_loader = modeltrn.test_data_loader
device = modeltrn.device

train.train(model=model, optimizer=optimizer, \
            loss_fn=loss_fn, train_loader=train_data_loader, \
            val_loader=test_data_loader, epochs=20, device=device)




