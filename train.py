import torch
import torch.nn as nn

from torch.utils.data import random_split, Dataloader
from torch.utils.tensorboard import SummaryWriter

from dataset import SimDataset
from model import ResNetAuto, ResNetAutoSteer

def training_loop(model, train_loader, val_loader, optimizer, scheduler, epochs, batches_before_log, device, writer):
    global_step = 0

    for epoch in range(epochs):
        print(f'Starting Epoch: {epoch + 1}...')

        total_train_loss_over_log = 0.0

        for batch_idx, batch_data in enumerate(train_loader):

            img, _, _, steering = batch_data

            img = img.to(device, non_blocking=True)
            img = img.permute(0, 3, 1, 2)
            steering = steering.to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(img.float())
            loss = nn.MSELoss()(output.float().squeeze(), steering.float().squeeze())

            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", loss.item(), global_step)
            total_train_loss_over_log += loss.item()
            global_step += 1

            if batch_idx % batches_before_log == batches_before_log - 1:
                train_loss_over_log = total_train_loss_over_log / batches_before_log
                print(f'Epoch: {epoch + 1}, Mini-Batches Completed: {(batch_idx + 1)},'
                      f' Train Loss: {train_loss_over_log:.6f}')

                total_train_loss_over_log = 0.0

        model.eval()
        with torch.no_grad():
            num_val_batches = 0
            total_val_loss = 0.0
            for val_batch_idx, val_batch_data in enumerate(val_loader):
                num_val_batches += 1

                img_val, _, _, steering_val = val_batch_data

                img_val = img_val.to(device, non_blocking=True)
                img_val = img_val.permute(0, 3, 1, 2)
                steering_val = steering_val.to(device, non_blocking=True)

                output_val = model(img_val.float())

                val_loss = nn.MSELoss()(output_val.float().squeeze(), steering_val.float().squeeze())

                total_val_loss += val_loss.item()

            val_loss_over_epoch = total_val_loss / num_val_batches
            writer.add_scalar("Loss/val", val_loss_over_epoch, epoch+1)
            print(f"Epoch {epoch + 1} is completed! Val loss: {val_loss_over_epoch:.6f}")

        model.train()
        scheduler.step(val_loss_over_epoch)
        print('Learning rate: {0}'.format(optimizer.param_groups[0]['lr']))

        torch.save(model.state_dict(), f"models/ResNetSteer_v5_{epoch + 1}.pth")
        print("Net Saved")

    print("Finished Training")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = r""
    dataset = SimDataset(data_path)

    train_size = int(0.9 * len(dataset))
    val_size = int((len(dataset) - train_size))

    print("Train size: ", train_size)
    print("Val size: ", val_size)

    train_set, val_set = random_split(dataset, (train_size, val_size))

    generator = torch.manual_seed(42)
    train_dataloader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=0, generator=generator)
    val_dataloader = DataLoader(dataset=val_set, batch_size=16, shuffle=False, num_workers=0, generator=generator)

    model = ResNetAutoSteer()
    model.to(device)
    print("Model init.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.2)

    writer = SummaryWriter()
    training_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, 30, 99999, device, writer)
