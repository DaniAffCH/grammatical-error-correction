import logging


def trainOneStep(model, dataloader, optimizer, lr_scheduler, criterion, logger, device):
    total_loss = 0
    for input, tgt_in, tgt_out in dataloader:
        out = model(input["input_ids"].to(device), tgt_in["input_ids"].to(device),
                    input["attention_mask"].to(device), tgt_in["attention_mask"].to(device))

        # BUG?????
        out = out.view(-1, out.size(2))
        gt = tgt_out["input_ids"].contiguous().view(-1).to(device)

        # Not sure though
        loss = criterion(out, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    logger.log(logging.INFO, f"Avg loss:{total_loss/len(dataloader)}")
