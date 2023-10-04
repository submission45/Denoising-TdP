import torch
from tqdm import tqdm

def load_model_tdp(checkpoints_dir, device, model_dir, **kwargs):
    from collections import OrderedDict

    from models.tdp.nn.tdp import TdPModel
    parameters = {
        "dropout_rate": 0.4,
        "encoder_pool_type": "max",
        "layers": 6,
        "compression": 1,
        "bottleneck": False,
        "pool_steps": [2, 2, 2, 2, 2, 2, 2],
        "activation": {
            "name": "relu",
            "args": {}
            }
        }


    path = '{}/checkpoints/{}/model_best.pt'.format(checkpoints_dir, model_dir)
    print(path)
    model = TdPModel(**parameters, verbose=False, n_classes=2)

    if torch.cuda.is_available():
        state_dict = torch.load(path)['model_state_dict']
        try:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
        model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_logits_return_labels_and_predictions(model,
                                                 dataloader, device=torch.device("cpu"),
                                                 *args, **kwargs):
    import torch.nn.functional as trchfnctnl
    """
    compute the logits given input data loader and model
    :param model: model utilized for the logits computation
    :param dataloader: loader for the training data
    :param device: device used for computation
    :return: logits and targets
    """
    logits = []
    labels = []
    hard_predictions = []
    soft_predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, ascii=True, ncols=50, colour='red')):

            data = data.unsqueeze(dim=1)
            preds_logit = model(data.to(device))
            logits.append(preds_logit.detach().cpu())

            if preds_logit.shape[1] >= 2:
                soft_prob = trchfnctnl.softmax(preds_logit, dim=1) #preds_logit #trchfnctnl.softmax(preds_logit, dim=1)
                preds = torch.argmax(soft_prob, dim=1)
            else:
                soft_prob = trchfnctnl.sigmoid(preds_logit) #preds_logit #
                preds = torch.round(soft_prob)

            hard_predictions.append(preds.detach().cpu().reshape(-1, 1))
            labels.append(target.detach().cpu().reshape(-1, 1))
            soft_predictions.append(soft_prob.detach().cpu())

    logits = torch.vstack(logits)
    labels = torch.vstack(labels).reshape(-1)
    hard_predictions = torch.vstack(hard_predictions).reshape(-1)
    soft_predictions = torch.vstack(soft_predictions)

    return logits, labels, hard_predictions, soft_predictions

if __name__ == '__main__':
    model = load_model_tdp(checkpoints_dir=".", device=1, model_dir='tdp_hb')

