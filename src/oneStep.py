from _utils import moveToDevice, masked_accuracy
import torch


def trainOneStep(model, dataloader, device, optimizer, criterion, tokenizer, lr_scheduler):
    model.train()
    total_loss = 0
    total_acc = 0
    for src, trg in dataloader:
        optimizer.zero_grad()

        src = moveToDevice(src, device)
        trg = moveToDevice(trg, device)

        trg_in = {
            "input_ids": trg["input_ids"][:, :-1],
            "attention_mask": trg["attention_mask"][:, :-1]
        }

        trg_out = trg["input_ids"][:, 1:]

        y_hat = model((src, trg_in))

        y_hat = y_hat.view(-1, y_hat.size(2))
        y = trg_out.contiguous().view(-1)

        loss = criterion(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = masked_accuracy(
            y, predicted, pad_idx=tokenizer.pad_token_id)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        total_acc += acc.item()

    return total_loss/len(dataloader), total_acc/len(dataloader)
