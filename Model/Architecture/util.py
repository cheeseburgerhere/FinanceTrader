
import numpy as np
import torch

def directional_loss(outputs, targets, sequences):
    # Calculate the direction of predictions and actual values
    pred_dir = (outputs.squeeze() - sequences[:, -1, 3]) >= 0
    true_dir = (targets - sequences[:, -1, 3]) >= 0
    # Penalize mismatched directions
    mismatches = (pred_dir != true_dir).float()
    return mismatches.mean()


#currenty not available will need a rewrite
# def inverseScale(sequence, scaler, name, predIndex, dataIndex):
#     placeHolder=np.zeros((batch_size,lookback_window,input_size))
#     placeHolder[:,-1,dataIndex]=sequence.cpu().numpy()
#     reverseScaled=scaler.inverse_transform(placeHolder[predIndex,:,:])
#     return zScalerDic[name].inverse_transform(reverseScaled)

def asymmetric_huber_loss(y_pred, y_true, delta=.5, alpha=3.0, beta=3.0):
    error = y_pred - y_true
    loss = torch.where(
        error < -delta,
        alpha * 0.5 * (error ** 2),  # Underprediction penalty
        torch.where(
            error > delta,
            beta * 0.5 * (error ** 2),  # Overprediction penalty
            delta * torch.abs(error) - 0.5 * delta ** 2  # Standard Huber region
        )
    )
    return torch.mean(loss)