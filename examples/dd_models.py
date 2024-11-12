import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from pyEM.math import softmax, calc_fval

def simulate_ddm(params, nblocks=3, ntrials=24):
    drift, thresh, ndt = params
    noise = 1.0  # fixed noise for simplicity
    dt = 0.01
    
    choices = np.zeros((nblocks, ntrials))
    response_times = np.zeros((nblocks, ntrials))
    
    for b in range(nblocks):
        for t in range(ntrials):
            # Initialize variables
            x = 0
            t_time = 0
            
            # Accumulate evidence
            while abs(x) < thresh:
                x += drift * dt + noise * np.sqrt(dt) * np.random.randn()
                t_time += dt
                
            # Determine decision
            choices[b, t] = 1 if x >= thresh else 0
            response_times[b, t] = t_time + ndt
            
    return choices, response_times

def fit(params, choices, response_times, prior=None, output='npl'):

    drift, thresh, ndt = params
    noise = 1.0  # fixed noise for simplicity
    dt = 0.01
    
    nblocks, ntrials = choices.shape
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    negll  = 0
    for b in range(nblocks):
        for t in range(ntrials):
            this_choice = choices[t]
            rt = response_times[t] - ndt
            if rt < 0:
                return -np.inf  # invalid non-decision time
            
            # Compute drift diffusion probability using approximations
            p_up = 1 / (1 + np.exp(-2 * drift * thresh / (noise**2)))
            ch_prob[b, t, 0] = p_up
            ch_prob[b, t, 1] = 1 - p_up
            p_correct = p_up if this_choice == 1 else 1 - p_up
            negll += -np.log(p_correct)
        
    # CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR (OR NEGLL)
    if (output == 'npl') or (output == 'nll'):
        fval = calc_fval(negll, params, prior=prior, output=output)
        return fval
    
    elif output == 'all':
        subj_dict = {'params'     : [drift, thresh, ndt],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'response_times'    : response_times, 
                     'pe'         : pe, 
                     'negll'      : negll}
        return subj_dict