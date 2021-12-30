from model import SigNet, ContrastiveLoss
import os
from data import get_data_loader
from PIL import ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from metrics import accuracy
from argparse import ArgumentParser
import warnings
from tensorboardX import SummaryWriter

seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))
writer = SummaryWriter(log_dir='scalar')

def train(model, optimizer, criterion, dataloader,testloader,epoch, log_interval=100):
    model.train()
    running_loss = 0
    number_samples = 0
    iter=epoch*380
    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        loss.backward()
        optimizer.step()

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        iter+=1
        writer.add_scalar('/train_loss', running_loss/number_samples, iter)
        
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))
            loss, acc = eval(model, criterion, testloader)
            writer.add_scalar('/accuracy', acc, iter)
            running_loss = 0
            number_samples = 0
            

@torch.no_grad()
def eval(model, criterion, dataloader, log_interval=50):
    model.eval()
    running_loss = 0
    number_samples = 0

    distances = []

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy = accuracy(distances, y)
    print(f'Max accuracy: {max_accuracy}')
    return running_loss / number_samples, max_accuracy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dataset', type=str, choices=['cedar', 'bengali', 'hindi', 'svc2004'], default='cedar')
    args = parser.parse_args()
    print(args)

    model = SigNet().to(device)
    criterion = ContrastiveLoss(alpha=1, beta=1, margin=1).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    num_epochs = 20

    image_transform = transforms.Compose([
        transforms.Resize((155, 220)),
        ImageOps.invert,
        transforms.ToTensor(),
        # TODO: add normalize
    ])

    trainloader = get_data_loader(is_train=True, batch_size=args.batch_size, image_transform=image_transform, dataset=args.dataset)
    testloader = get_data_loader(is_train=False, batch_size=args.batch_size, image_transform=image_transform, dataset=args.dataset)
    os.makedirs('checkpoints', exist_ok=True)

    model.train()
    print(model)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Training', '-'*20)
        train(model, optimizer, criterion, trainloader,testloader,epoch)
        print('Evaluating', '-'*20)
        loss, acc = eval(model, criterion, testloader)
        scheduler.step()

        to_save = {
            'model': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optim': optimizer.state_dict(),
        }

        print('Saving checkpoint..')
        torch.save(to_save, 'checkpoints/epoch_{}_loss_{:.3f}_acc_{:.3f}.pt'.format(epoch, loss, acc))
    writer.close()
    print('Done')
