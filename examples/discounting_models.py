import numpy as np
from tqdm import tqdm
import sys
from copy import deepcopy
sys.path.append('../')
from pyEM.math import softmax, norm2beta, norm2alpha

def hyperbolic(k, N, m_self, logk=False):
    '''
    convert m to utility based on social distance N using hyperbolic model
    v_observed~v0/(1+exp(logk)*N)
    ''' 
    if logk:
        v_other = (m_self) / (1 + np.exp(np.log(k))*N)
    else:
        v_other = (m_self) / (1 + k*N)

    return v_other

def simulate(params, options=[[155, 145, 135, 125, 115, 105, 95, 85, 75],
                              [ 75,  75,  75,  75,  75,  75, 75, 75, 75]], N=[1,2,5,10,20,50,100]):
    """
    Simulate the basic RW model.

    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """
    nsubjects   = params.shape[0]
    ntrials     = len(options[0])
    nblocks     = len(N)

    ev          = np.zeros((nsubjects, nblocks, ntrials))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object') # 0 selfish, 1 generous
    choices_G   = np.zeros((nsubjects, nblocks, ntrials,))
    outcomes    = np.zeros((nsubjects, nblocks, ntrials, 2)) # self, other
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    for subj_idx in tqdm(range(nsubjects)):
        beta, k = params[subj_idx,:]
            
        for b in range(nblocks): 
            for t in range(ntrials):
                # calculate utility of choices
                ev[subj_idx, b, t] = hyperbolic(k, N[b], options[0][t]-options[1][t])

                # calculate generous choice probability
                ch_prob[subj_idx, b, t] = 1 / (1 + np.exp(-beta * ev[subj_idx, b, t]))

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['Selfish', 'Generous'], 
                                                size=1, 
                                                p=[1-ch_prob[subj_idx, b, t], ch_prob[subj_idx, b, t]])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'Generous':
                    c = 1
                    choices_G[subj_idx, b, t] = 1
                    outcomes[subj_idx, b, t, :]   = [deepcopy(options[1][t]), deepcopy(options[1][t])]
                    choice_nll[subj_idx, b, t] = ch_prob[subj_idx, b, t].copy()
                else:
                    c = 0
                    choices_G[subj_idx, b, t] = 0
                    outcomes[subj_idx, b, t, :] = [deepcopy(options[0][t]), 0]
                    choice_nll[subj_idx, b, t] = 1 - ch_prob[subj_idx, b, t].copy()                

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_G' : choices_G, 
                 'outcomes'  : outcomes, 
                 'choice_nll': choice_nll}

    return subj_dict

def fit(params, choices, rewards, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with "A" or "B" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    beta = norm2beta(params[0])
    lr   = norm2alpha(params[1])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = rewards.shape

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:]    = [.5, .5]

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
            else:
                c = 1
                choices_A[b, t] = 0

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = rewards[b, t] - ev[b, t, c]

            # update EV
            ev[b, t+1, :] = ev[b, t, :].copy()
            ev[b, t+1, c] = ev[b, t, c] + (lr * pe[b, t])
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [beta, lr],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_A'  : choices_A, 
                     'rewards'    : rewards, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict