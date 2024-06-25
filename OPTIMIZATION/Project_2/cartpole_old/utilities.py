import numpy as np

def EWMA(past_value, current_value, alpha):
    return (1 - alpha) * past_value + alpha * current_value

def calc_ewma(values, period):
    alpha = 2 / (period + 1)
    result = []
    for v in values:
        try:
            prev_value = result[-1]
        except IndexError:
            prev_value = 0

        new_value = EWMA(prev_value, v, alpha)
        result.append(new_value)
    return np.array(result)
    
def correction(averaged_value, beta, steps):
    return averaged_value / (1 - (beta ** steps))

def calc_corrected_ewma(values, period):
    ewma = calc_ewma(values, period)
    
    alpha = 2 / (period + 1)
    beta = 1 - alpha
    
    result = []
    for step, v in enumerate(ewma):
        adj_value = correction(v, beta, step + 1)
        result.append(adj_value)
        
    return np.array(result)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)