import torch.nn.functional as F
import torch

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            input,target = batch
            print(input)
            print(target)
            print(device)
            input = input.to(device)
            print("after input")
            target = target.to(device)
            print("after output")
            output = model(input)
            print("after model")
            loss = loss_fn(output,target)
            print("loss input")
            loss.backward()
            optimizer.step()
            training_loss+= loss.data.item()
            
        training_loss /= len(train_loader)
        
        model.eval()
        num_correct = 0
        num_examples = 0
        
        for batch in val_loader:
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output,target)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output),dim=1)[1],target).view(-1)
            
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)
        
        print('Epoch: {}, Training loss: {:.2f}, \
            Validation Loss: {:.2f}, \
                accuracy: {:.2f}'.format(epoch, training_loss, valid_loss, num_correct/num_examples)
            )
    print("here")
    torch.save(model.state_dict(),"model/net/final.model")