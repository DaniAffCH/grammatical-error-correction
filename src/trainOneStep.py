import logging


def trainOneStep(model, dataloader, optimizer, lr_scheduler, criterion, device):
    total_loss = 0
    for input, tgt_in, tgt_out in dataloader:
        optimizer.zero_grad()

        out = model(input["input_ids"].to(device), tgt_in["input_ids"].to(device),
                    input["attention_mask"].to(device), tgt_in["attention_mask"].to(device))

        # BUG?????
        out = out.view(-1, out.size(2))
        gt = tgt_out["input_ids"].contiguous().view(-1).to(device)

        # Not sure though
        loss = criterion(out, gt)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    return total_loss/len(dataloader)
