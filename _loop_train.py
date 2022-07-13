import torch
import torch.nn as nn


def _accumulate(s_model, t_model, loader, optimizer, get_metrics, 
                device, metrics, criterion):
    """
    Args:
            s_model: primary model
            t_model: auxiliary model
    """
    s_model.train()
    metrics.reset()
    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        weights = labels.detach().cpu().numpy().sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])
        weights = torch.tensor([weights, 1-weights], dtype=torch.float32).to(device)
        criterion.update_weight(weight=weights)

        optimizer.zero_grad()

        s_outputs = s_model(images)
        probs = nn.Softmax(dim=1)(s_outputs)
        preds = torch.max(probs, 1)[1].detach().cpu().numpy()
        t_outputs = t_model(images)

        loss = criterion(s_outputs, t_outputs, labels)
        loss.backward()

        optimizer.step()
        
        if get_metrics:
            metrics.update(labels.detach().cpu().numpy(), preds)
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss