# ----------------------------------------
# 1Q–4α–1β  SIMULATE (Rhoads et al., 2025)
# ----------------------------------------
def gen_rnd_blocks(items, nblocks=2, nsubjects=100):
    perms = list(permutations(items))
    for _ in range(nsubjects):
        # Randomly pick nblocks permutations with replacement
        blocks = random.choices(perms, k=nblocks)
        combined = tuple(chain.from_iterable(blocks))
        yield combined

def rw4a1b_sim(params: np.ndarray,
                    nblocks: int = 12,
                    ntrials: int = 20):
    """
    Simulate a 4-option 1Q RW with one beta and four learning rates:
      a_self_pos, a_self_neg, a_other_pos, a_other_neg

    Each trial shows a PAIR OF OPTIONS (indices 0..3) and the agent picks one of those two.
    Outcomes for SELF and OTHER are drawn independently from option-specific marginals over {-1,0,+1}

    Fixed design:
      - There are 6 unique option-pairs from {0,1,2,3}: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
      - For 12 blocks, we cycle those 6 pairs twice (blocks 0..11).
      - Each block also has a fixed pattern type for outcome marginals, cycled over 4 types:
            (+/+), (+/-), (-/+), (-/-)  and then repeat.
        Pattern type is fixed within a block.

    Returns:
      - choices          : (S,B,T) int indices 0..3 (chosen option among the shown pair)
      - outcomes_self    : (S,B,T) int  in {-1,0,+1}
      - outcomes_other   : (S,B,T) int  in {-1,0,+1}
      - option_pairs     : (S,B,T,2) int indices for the two shown options on each trial
      - also EV, ch_prob (over 4), and PE components
    """
    assert nblocks % 6 == 0, "nblocks should be multiple of 6 for full counterbalancing"
    
    # Bounds check
    beta_all, a_self_pos_all, a_self_neg_all, a_other_pos_all, a_other_neg_all = (params[:, i].astype(float) for i in range(5))
    if not ((beta_all > 1e-5) & (beta_all <= 20.0)).all():
        raise ValueError("beta out of bounds")
    for arr, name in [(a_self_pos_all, "a_self_pos"), (a_self_neg_all, "a_self_neg"),
                      (a_other_pos_all, "a_other_pos"), (a_other_neg_all, "a_other_neg")]:
        if not ((0.0 <= arr) & (arr <= 1.0)).all():
            raise ValueError(f"{name} out of bounds")

    rng = np.random.default_rng()
    nsubjects = params.shape[0]

    # Outputs 
    choices        = np.zeros((nsubjects, nblocks, ntrials), dtype=object) # A,B,C,D
    outcomes_self  = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    outcomes_other = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    option_pairs   = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=object)

    # Optional diagnostics
    EV           = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nsubjects, nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    # create task structure
    all_pairs = ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    block_orders = list(gen_rnd_blocks(['AB', 'AC', 'AD', 'BC', 'BD', 'CD'], 
                        nblocks=nblocks, nsubjects=nsubjects))
    if ntrials == 20:
        # high 75%, mid 15%, low 10%
        opt_templates = {'+': [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, -1, -1],
                         '-': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,  1,  1]}
    else: # get proportion of good (1,0,-1) and bad (-1,0,1) at high 75%, mid 15%, low 10%
        opt_templates = {'+': np.random.choice([1, 0, -1], size=ntrials, p=[0.75, 0.15, 0.10]),
                         '-': np.random.choice([-1, 0, 1], size=ntrials, p=[0.75, 0.15, 0.10])}
    opt_types = {'A': ('+','+'), 'B': ('+','-'),'C': ('-','+'), 'D': ('-','-')}

    for s in range(nsubjects):
        beta        = beta_all[s]
        a_self_pos  = a_self_pos_all[s]
        a_self_neg  = a_self_neg_all[s]
        a_other_pos = a_other_pos_all[s]
        a_other_neg = a_other_neg_all[s]

        for b in range(nblocks):
            # get block pair
            opt1, opt2 = block_orders[s][b]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # create possible outcomes for this block
            all_outcomes = {'A_self' :[np.nan]*ntrials, 'A_other':[np.nan]*ntrials,
                            'B_self' :[np.nan]*ntrials, 'B_other':[np.nan]*ntrials,
                            'C_self' :[np.nan]*ntrials, 'C_other':[np.nan]*ntrials,
                            'D_self' :[np.nan]*ntrials, 'D_other':[np.nan]*ntrials
                            }
            for this_opt in (opt1, opt2):
                self_kind, other_kind = opt_types[this_opt]
                all_outcomes[f'{this_opt}_self']  = rng.permutation(opt_templates[self_kind])
                all_outcomes[f'{this_opt}_other'] = rng.permutation(opt_templates[other_kind])
            
            EV[s, b, 0, :] = 0
            for t in range(ntrials):
                # the two shown options on this trial (fixed per block)
                option_pairs[s, b, t, 0] = opt1
                option_pairs[s, b, t, 1] = opt2
                
                # softmax over the two shown options
                shown_vals = np.array([EV[s, b, t, o1], EV[s, b, t, o2]], dtype=float)
                p = softmax(shown_vals, beta)
                ch_prob[s, b, t, o1] = p[0]
                ch_prob[s, b, t, o2] = p[1]
                choices[s, b, t] = rng.choice([opt1, opt2], p=p)
                c = letter_to_idx[choices[s, b, t]]

                # get outcomes from choices and all_outcomes
                outcomes_self[s, b, t] = all_outcomes[f'{choices[s, b, t]}_self'][t]
                outcomes_other[s, b, t] = all_outcomes[f'{choices[s, b, t]}_other'][t]

                # compute prediction errors
                pe_self[s, b, t] = outcomes_self[s, b, t] - EV[s, b, t, c]
                pe_other[s, b, t] = outcomes_other[s, b, t] - EV[s, b, t, c]

                pe_self_pos[s, b, t]  = pe_self[s, b, t] if pe_self[s, b, t]   >= 0.0 else 0.0
                pe_self_neg[s, b, t]  = pe_self[s, b, t] if pe_self[s, b, t]   <  0.0 else 0.0
                pe_other_pos[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] >= 0.0 else 0.0
                pe_other_neg[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] <  0.0 else 0.0

                # update the chosen option
                EV[s, b, t+1, :] = EV[s, b, t, :].copy()
                EV[s, b, t+1, c] = EV[s, b, t, c] + (a_self_pos  * pe_self_pos[s, b, t] + 
                                                     a_self_neg  * pe_self_neg[s, b, t] + 
                                                     a_other_pos * pe_other_pos[s, b, t] + 
                                                     a_other_neg * pe_other_neg[s, b, t])

    return {"params": params,
            "choices": choices,                # chosen option indices (A,B,C,D)
            "outcomes_self": outcomes_self,    # -1/0/+1
            "outcomes_other": outcomes_other,  # -1/0/+1
            "option_pairs": option_pairs,      # which two options were shown on each trial
            "EV": EV,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
            }

# ----------------------------------
# 1Q–4α–1β FIT (Rhoads et al., 2025)
# ----------------------------------
def rw4a1b_fit(params: np.ndarray,
               choices: np.ndarray,        # (B,T) chosen options (A,B,C,D)
               outcomes_self: np.ndarray,  # (B,T) in {-1,0,+1}
               outcomes_other: np.ndarray, # (B,T) in {-1,0,+1}
               option_pairs: np.ndarray,   # (B,T,2) indices of shown options per trial
               prior=None,
               output: str = "npl"):

    beta        = norm2beta(params[0])
    a_self_pos  = norm2alpha(params[1])
    a_self_neg  = norm2alpha(params[2])
    a_other_pos = norm2alpha(params[3])
    a_other_neg = norm2alpha(params[4])

    # Bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    for a in (a_self_pos, a_self_neg, a_other_pos, a_other_neg):
        if not (0.0 <= a <= 1.0):
            return 1e7

    # Convert choices (accepts letters or indices)
    choices_arr = np.asarray(choices)
    if not np.issubdtype(choices_arr.dtype, np.number):
        letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
        choices_arr = np.vectorize(letter_to_idx.get)(choices_arr)
    choices_arr = choices_arr.astype(int, copy=False)

    nblocks, ntrials = outcomes_self.shape
    EV           = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nblocks, ntrials), dtype=float)

    nll = 0.0
    for b in range(nblocks):
        EV[b, 0, :] = 0.0
        for t in range(ntrials):
            # get shown options
            opt1, opt2 = option_pairs[b, t, 0], option_pairs[b, t, 1]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # get probability of the chosen option
            c = letter_to_idx[choices[b, t]]
            shown_vals = np.array([EV[b, t, o1], EV[b, t, o2]], dtype=float)
            probs_two  = softmax(shown_vals, beta)  # len=2
            ch_prob[b, t, o1] = probs_two[0]
            ch_prob[b, t, o2] = probs_two[1]
            nll += -np.log(probs_two[0] if c == o1 else probs_two[1] + 1e-12)

            # compute prediction errors
            pe_self[b, t]  = outcomes_self[b, t]  - EV[b, t, c]
            pe_other[b, t] = outcomes_other[b, t] - EV[b, t, c]

            pe_self_pos[b, t]  = pe_self[b, t]  if pe_self[b, t]  >= 0.0 else 0.0
            pe_self_neg[b, t]  = pe_self[b, t]  if pe_self[b, t]  <  0.0 else 0.0
            pe_other_pos[b, t] = pe_other[b, t] if pe_other[b, t] >= 0.0 else 0.0
            pe_other_neg[b, t] = pe_other[b, t] if pe_other[b, t] <  0.0 else 0.0

            # update chosen option
            EV[b, t+1, :] = EV[b, t, :].copy()
            EV[b, t+1, c] = EV[b, t, c] + (a_self_pos  * pe_self_pos[b, t] +
                                           a_self_neg  * pe_self_neg[b, t] +
                                           a_other_pos * pe_other_pos[b, t] +
                                           a_other_neg * pe_other_neg[b, t])

    if output == "all":
        return {
            "params": [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg],
            "choices": choices_arr,
            "outcomes_self": outcomes_self,
            "outcomes_other": outcomes_other,
            "option_pairs": option_pairs,
            "EV": EV,
            "nll": nll,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
        }
    else:
        return calc_fval(nll, params, prior=prior, output=output)



# ----------------------------------------
# 1Q–4α–1β  SIMULATE → now 2Q (EV_self, EV_other)
# ----------------------------------------
from itertools import permutations, chain
import numpy as np, random

def gen_rnd_blocks(items, nblocks=2, nsubjects=100):
    perms = list(permutations(items))
    for _ in range(nsubjects):
        # Randomly pick nblocks permutations with replacement
        blocks = random.choices(perms, k=nblocks)
        combined = tuple(chain.from_iterable(blocks))
        yield combined

def softmax(vals2, beta):
    # expects len(vals2)==2
    x = np.array(vals2, dtype=float) * beta
    x -= x.max()  # stabilize
    e = np.exp(x)
    return e / e.sum()

def rw4a1b_sim(params: np.ndarray,
               nblocks: int = 12,
               ntrials: int = 20):
    """
    Simulate a 4-option 2Q RW with one beta and four learning rates:
      a_self_pos, a_self_neg, a_other_pos, a_other_neg

    Decision policy: softmax over (EV_self + EV_other).
    Updates: EV_self updated from PE_self; EV_other updated from PE_other.

    Returns (backward compatible):
      - EV : sum of EV_self + EV_other
      - also EV_self and EV_other separately
    """
    assert nblocks % 6 == 0, "nblocks should be multiple of 6 for full counterbalancing"
    
    # Bounds check
    beta_all, a_self_pos_all, a_self_neg_all, a_other_pos_all, a_other_neg_all = (params[:, i].astype(float) for i in range(5))
    if not ((beta_all > 1e-5) & (beta_all <= 20.0)).all():
        raise ValueError("beta out of bounds")
    for arr, name in [(a_self_pos_all, "a_self_pos"), (a_self_neg_all, "a_self_neg"),
                      (a_other_pos_all, "a_other_pos"), (a_other_neg_all, "a_other_neg")]:
        if not ((0.0 <= arr) & (arr <= 1.0)).all():
            raise ValueError(f"{name} out of bounds")

    rng = np.random.default_rng()
    nsubjects = params.shape[0]

    # Outputs 
    choices        = np.zeros((nsubjects, nblocks, ntrials), dtype=object) # A,B,C,D
    outcomes_self  = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    outcomes_other = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    option_pairs   = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=object)

    # Diagnostics / values
    EV_self     = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)
    EV_other    = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)
    EV          = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)  # sum for compatibility
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    # create task structure
    all_pairs = ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    block_orders = list(gen_rnd_blocks(['AB', 'AC', 'AD', 'BC', 'BD', 'CD'], 
                        nblocks=nblocks, nsubjects=nsubjects))
    if ntrials == 20:
        # high 75%, mid 15%, low 10%
        opt_templates = {'+': [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, -1, -1],
                         '-': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,  1,  1]}
    else: # get proportion of good (1,0,-1) and bad (-1,0,1) at high 75%, mid 15%, low 10%
        opt_templates = {'+': np.random.choice([1, 0, -1], size=ntrials, p=[0.75, 0.15, 0.10]),
                         '-': np.random.choice([-1, 0, 1], size=ntrials, p=[0.75, 0.15, 0.10])}
    opt_types = {'A': ('+','+'), 'B': ('+','-'),'C': ('-','+'), 'D': ('-','-')}

    for s in range(nsubjects):
        beta        = beta_all[s]
        a_self_pos  = a_self_pos_all[s]
        a_self_neg  = a_self_neg_all[s]
        a_other_pos = a_other_pos_all[s]
        a_other_neg = a_other_neg_all[s]

        for b in range(nblocks):
            # get block pair
            opt1, opt2 = block_orders[s][b]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # create possible outcomes for this block
            all_outcomes = {'A_self' :[np.nan]*ntrials, 'A_other':[np.nan]*ntrials,
                            'B_self' :[np.nan]*ntrials, 'B_other':[np.nan]*ntrials,
                            'C_self' :[np.nan]*ntrials, 'C_other':[np.nan]*ntrials,
                            'D_self' :[np.nan]*ntrials, 'D_other':[np.nan]*ntrials
                            }
            for this_opt in (opt1, opt2):
                self_kind, other_kind = opt_types[this_opt]
                all_outcomes[f'{this_opt}_self']  = rng.permutation(opt_templates[self_kind])
                all_outcomes[f'{this_opt}_other'] = rng.permutation(opt_templates[other_kind])
            
            # initialize values
            EV_self[s, b, 0, :]  = 0.0
            EV_other[s, b, 0, :] = 0.0
            EV[s, b, 0, :]       = 0.0

            for t in range(ntrials):
                # the two shown options on this trial (fixed per block)
                option_pairs[s, b, t, 0] = opt1
                option_pairs[s, b, t, 1] = opt2
                
                # softmax over the two shown options, using sum of EVs
                shown_vals = np.array([EV_self[s, b, t, o1] + EV_other[s, b, t, o1],
                                       EV_self[s, b, t, o2] + EV_other[s, b, t, o2]], dtype=float)
                p = softmax(shown_vals, beta)
                ch_prob[s, b, t, o1] = p[0]
                ch_prob[s, b, t, o2] = p[1]
                choices[s, b, t] = rng.choice([opt1, opt2], p=p)
                c = letter_to_idx[choices[s, b, t]]

                # get outcomes from choices and all_outcomes
                outcomes_self[s, b, t]  = all_outcomes[f'{choices[s, b, t]}_self'][t]
                outcomes_other[s, b, t] = all_outcomes[f'{choices[s, b, t]}_other'][t]

                # compute prediction errors against their own Qs
                pe_self[s, b, t]  = outcomes_self[s, b, t]  - EV_self[s,  b, t, c]
                pe_other[s, b, t] = outcomes_other[s, b, t] - EV_other[s, b, t, c]

                pe_self_pos[s, b, t]  = pe_self[s, b, t]  if pe_self[s, b, t]  >= 0.0 else 0.0
                pe_self_neg[s, b, t]  = pe_self[s, b, t]  if pe_self[s, b, t]  <  0.0 else 0.0
                pe_other_pos[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] >= 0.0 else 0.0
                pe_other_neg[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] <  0.0 else 0.0

                # update the chosen option (each Q updated by its own PE/α)
                EV_self[s,  b, t+1, :] = EV_self[s,  b, t, :].copy()
                EV_other[s, b, t+1, :] = EV_other[s, b, t, :].copy()

                EV_self[s,  b, t+1, c]  = EV_self[s,  b, t, c]  + (a_self_pos  * pe_self_pos[s, b, t] +
                                                                    a_self_neg  * pe_self_neg[s, b, t])
                EV_other[s, b, t+1, c]  = EV_other[s, b, t, c]  + (a_other_pos * pe_other_pos[s, b, t] +
                                                                    a_other_neg * pe_other_neg[s, b, t])

                # keep the summed EV for compatibility/debug
                EV[s, b, t+1, :] = EV_self[s, b, t+1, :] + EV_other[s, b, t+1, :]

    return {"params": params,
            "choices": choices,                # chosen options (A,B,C,D)
            "outcomes_self": outcomes_self,    # -1/0/+1
            "outcomes_other": outcomes_other,  # -1/0/+1
            "option_pairs": option_pairs,      # shown options
            "EV": EV,                          # sum (compatibility)
            "EV_self": EV_self,
            "EV_other": EV_other,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
            }

# ----------------------------------
# 1Q–4α–1β FIT → now 2Q (EV_self, EV_other)
# ----------------------------------
def norm2beta(x):   # unchanged helper (placeholder)
    return float(x)

def norm2alpha(x):  # unchanged helper (placeholder)
    return float(x)

def calc_fval(nll, params, prior=None, output="npl"):
    # unchanged helper (placeholder)
    return nll

def rw4a1b_fit(params: np.ndarray,
               choices: np.ndarray,        # (B,T) chosen options (A,B,C,D)
               outcomes_self: np.ndarray,  # (B,T) in {-1,0,+1}
               outcomes_other: np.ndarray, # (B,T) in {-1,0,+1}
               option_pairs: np.ndarray,   # (B,T,2) shown options per trial (letters)
               prior=None,
               output: str = "npl"):

    beta        = norm2beta(params[0])
    a_self_pos  = norm2alpha(params[1])
    a_self_neg  = norm2alpha(params[2])
    a_other_pos = norm2alpha(params[3])
    a_other_neg = norm2alpha(params[4])

    # Bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    for a in (a_self_pos, a_self_neg, a_other_pos, a_other_neg):
        if not (0.0 <= a <= 1.0):
            return 1e7

    # Maps
    letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}

    # Convert choices (accepts letters or indices)
    choices_arr = np.asarray(choices)
    if not np.issubdtype(choices_arr.dtype, np.number):
        choices_arr = np.vectorize(letter_to_idx.get)(choices_arr)
    choices_arr = choices_arr.astype(int, copy=False)

    nblocks, ntrials = outcomes_self.shape
    EV_self     = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    EV_other    = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    EV_sum      = np.zeros((nblocks, ntrials + 1, 4), dtype=float)  # for compatibility
    ch_prob      = np.zeros((nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nblocks, ntrials), dtype=float)

    nll = 0.0
    for b in range(nblocks):
        EV_self[b,  0, :] = 0.0
        EV_other[b, 0, :] = 0.0
        EV_sum[b,   0, :] = 0.0
        for t in range(ntrials):
            # get shown options (letters expected here)
            opt1, opt2 = option_pairs[b, t, 0], option_pairs[b, t, 1]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # chosen option index
            c = letter_to_idx[choices[b, t]] if not isinstance(choices[b, t], (int, np.integer)) else int(choices[b, t])

            # probability of the chosen option based on sum of Qs
            shown_vals = np.array([EV_self[b, t, o1] + EV_other[b, t, o1],
                                   EV_self[b, t, o2] + EV_other[b, t, o2]], dtype=float)
            probs_two  = softmax(shown_vals, beta)  # len=2
            ch_prob[b, t, o1] = probs_two[0]
            ch_prob[b, t, o2] = probs_two[1]
            nll += -np.log((probs_two[0] if c == o1 else probs_two[1]) + 1e-12)

            # prediction errors against their respective Qs
            pe_self[b, t]  = outcomes_self[b, t]  - EV_self[b,  t, c]
            pe_other[b, t] = outcomes_other[b, t] - EV_other[b, t, c]

            pe_self_pos[b, t]  = pe_self[b, t]  if pe_self[b, t]  >= 0.0 else 0.0
            pe_self_neg[b, t]  = pe_self[b, t]  if pe_self[b, t]  <  0.0 else 0.0
            pe_other_pos[b, t] = pe_other[b, t] if pe_other[b, t] >= 0.0 else 0.0
            pe_other_neg[b, t] = pe_other[b, t] if pe_other[b, t] <  0.0 else 0.0

            # update chosen option in each Q separately
            EV_self[b,  t+1, :] = EV_self[b,  t, :].copy()
            EV_other[b, t+1, :] = EV_other[b, t, :].copy()

            EV_self[b,  t+1, c] = EV_self[b,  t, c] + (a_self_pos  * pe_self_pos[b, t] +
                                                       a_self_neg  * pe_self_neg[b, t])
            EV_other[b, t+1, c] = EV_other[b, t, c] + (a_other_pos * pe_other_pos[b, t] +
                                                       a_other_neg * pe_other_neg[b, t])

            EV_sum[b, t+1, :] = EV_self[b, t+1, :] + EV_other[b, t+1, :]

    if output == "all":
        return {
            "params": [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg],
            "choices": choices_arr,
            "outcomes_self": outcomes_self,
            "outcomes_other": outcomes_other,
            "option_pairs": option_pairs,
            "EV": EV_sum,          # sum (compatibility)
            "EV_self": EV_self,
            "EV_other": EV_other,
            "nll": nll,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
        }
    else:
        return calc_fval(nll, params, prior=prior, output=output)


# ----------------------------------------
# 2Q–4α–1β  SIMULATE (2Q: EV_self, EV_other)
# ----------------------------------------
def gen_rnd_blocks(items, nblocks=2, nsubjects=100):
    perms = list(permutations(items))
    for _ in range(nsubjects):
        # Randomly pick nblocks permutations with replacement
        blocks = random.choices(perms, k=nblocks)
        combined = tuple(chain.from_iterable(blocks))
        yield combined

def rw4a1b_sim(params: np.ndarray,
                    nblocks: int = 12,
                    ntrials: int = 20):
    """
    Simulate a 4-option RW with one beta and four learning rates:
      a_self_pos, a_self_neg, a_other_pos, a_other_neg

    Two separate value functions:
      - EV_self updated from PE_self
      - EV_other updated from PE_other

    Choice softmax uses EV_self + EV_other for the two shown options.
    """
    assert nblocks % 6 == 0, "nblocks should be multiple of 6 for full counterbalancing"
    
    # Bounds check
    beta_all, a_self_pos_all, a_self_neg_all, a_other_pos_all, a_other_neg_all = (params[:, i].astype(float) for i in range(5))
    if not ((beta_all > 1e-5) & (beta_all <= 20.0)).all():
        raise ValueError("beta out of bounds")
    for arr, name in [(a_self_pos_all, "a_self_pos"), (a_self_neg_all, "a_self_neg"),
                      (a_other_pos_all, "a_other_pos"), (a_other_neg_all, "a_other_neg")]:
        if not ((0.0 <= arr) & (arr <= 1.0)).all():
            raise ValueError(f"{name} out of bounds")

    rng = np.random.default_rng()
    nsubjects = params.shape[0]

    # Outputs 
    choices        = np.zeros((nsubjects, nblocks, ntrials), dtype=object) # A,B,C,D
    outcomes_self  = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    outcomes_other = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    option_pairs   = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=object)

    # Diagnostics
    EV_self      = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)
    EV_other     = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nsubjects, nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    # create task structure
    all_pairs = ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    block_orders = list(gen_rnd_blocks(['AB', 'AC', 'AD', 'BC', 'BD', 'CD'], 
                        nblocks=nblocks, nsubjects=nsubjects))
    if ntrials == 20:
        # high 75%, mid 15%, low 10%
        opt_templates = {'+': [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, -1, -1],
                         '-': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,  1,  1]}
    else:
        opt_templates = {'+': np.random.choice([1, 0, -1], size=ntrials, p=[0.75, 0.15, 0.10]),
                         '-': np.random.choice([-1, 0, 1], size=ntrials, p=[0.75, 0.15, 0.10])}
    opt_types = {'A': ('+','+'), 'B': ('+','-'),'C': ('-','+'), 'D': ('-','-')}

    for s in range(nsubjects):
        beta        = beta_all[s]
        a_self_pos  = a_self_pos_all[s]
        a_self_neg  = a_self_neg_all[s]
        a_other_pos = a_other_pos_all[s]
        a_other_neg = a_other_neg_all[s]

        for b in range(nblocks):
            # get block pair
            opt1, opt2 = block_orders[s][b]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # create possible outcomes for this block
            all_outcomes = {'A_self' :[np.nan]*ntrials, 'A_other':[np.nan]*ntrials,
                            'B_self' :[np.nan]*ntrials, 'B_other':[np.nan]*ntrials,
                            'C_self' :[np.nan]*ntrials, 'C_other':[np.nan]*ntrials,
                            'D_self' :[np.nan]*ntrials, 'D_other':[np.nan]*ntrials
                            }
            for this_opt in (opt1, opt2):
                self_kind, other_kind = opt_types[this_opt]
                all_outcomes[f'{this_opt}_self']  = rng.permutation(opt_templates[self_kind])
                all_outcomes[f'{this_opt}_other'] = rng.permutation(opt_templates[other_kind])
            
            EV_self[s,  b, 0, :] = 0.0
            EV_other[s, b, 0, :] = 0.0

            for t in range(ntrials):
                # the two shown options on this trial (fixed per block)
                option_pairs[s, b, t, 0] = opt1
                option_pairs[s, b, t, 1] = opt2
                
                # softmax over the two shown options, using sum of EVs
                shown_vals = np.array([EV_self[s, b, t, o1] + EV_other[s, b, t, o1],
                                       EV_self[s, b, t, o2] + EV_other[s, b, t, o2]], dtype=float)
                p = softmax(shown_vals, beta)
                ch_prob[s, b, t, o1] = p[0]
                ch_prob[s, b, t, o2] = p[1]
                choices[s, b, t] = rng.choice([opt1, opt2], p=p)
                c = letter_to_idx[choices[s, b, t]]

                # get outcomes from choices and all_outcomes
                outcomes_self[s, b, t]  = all_outcomes[f'{choices[s, b, t]}_self'][t]
                outcomes_other[s, b, t] = all_outcomes[f'{choices[s, b, t]}_other'][t]

                # compute prediction errors against their own Qs
                pe_self[s, b, t]  = outcomes_self[s, b, t]  - EV_self[s,  b, t, c]
                pe_other[s, b, t] = outcomes_other[s, b, t] - EV_other[s, b, t, c]

                pe_self_pos[s, b, t]  = pe_self[s, b, t]  if pe_self[s, b, t]  >= 0.0 else 0.0
                pe_self_neg[s, b, t]  = pe_self[s, b, t]  if pe_self[s, b, t]  <  0.0 else 0.0
                pe_other_pos[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] >= 0.0 else 0.0
                pe_other_neg[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] <  0.0 else 0.0

                # update the chosen option (each Q updated by its own PE/α)
                EV_self[s,  b, t+1, :] = EV_self[s,  b, t, :].copy()
                EV_other[s, b, t+1, :] = EV_other[s, b, t, :].copy()

                EV_self[s,  b, t+1, c]  = EV_self[s,  b, t, c]  + (a_self_pos  * pe_self_pos[s, b, t] +
                                                                    a_self_neg  * pe_self_neg[s, b, t])
                EV_other[s, b, t+1, c]  = EV_other[s, b, t, c]  + (a_other_pos * pe_other_pos[s, b, t] +
                                                                    a_other_neg * pe_other_neg[s, b, t])

    return {"params": params,
            "choices": choices,
            "outcomes_self": outcomes_self,
            "outcomes_other": outcomes_other,
            "option_pairs": option_pairs,
            "EV_self": EV_self,
            "EV_other": EV_other,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
            }

# ----------------------------------
# 2Q–4α–1β FIT (2Q: EV_self, EV_other)
# ----------------------------------
def rw4a1b_fit(params: np.ndarray,
               choices: np.ndarray,        # (B,T) chosen options (A,B,C,D)
               outcomes_self: np.ndarray,  # (B,T) in {-1,0,+1}
               outcomes_other: np.ndarray, # (B,T) in {-1,0,+1}
               option_pairs: np.ndarray,   # (B,T,2) indices of shown options per trial
               prior=None,
               output: str = "npl"):

    beta        = norm2beta(params[0])
    a_self_pos  = norm2alpha(params[1])
    a_self_neg  = norm2alpha(params[2])
    a_other_pos = norm2alpha(params[3])
    a_other_neg = norm2alpha(params[4])

    # Bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    for a in (a_self_pos, a_self_neg, a_other_pos, a_other_neg):
        if not (0.0 <= a <= 1.0):
            return 1e7

    # Convert choices (accepts letters or indices)
    choices_arr = np.asarray(choices)
    letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    if not np.issubdtype(choices_arr.dtype, np.number):
        choices_arr = np.vectorize(letter_to_idx.get)(choices_arr)
    choices_arr = choices_arr.astype(int, copy=False)

    nblocks, ntrials = outcomes_self.shape
    EV_self      = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    EV_other     = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nblocks, ntrials), dtype=float)

    nll = 0.0
    for b in range(nblocks):
        EV_self[b,  0, :] = 0.0
        EV_other[b, 0, :] = 0.0
        for t in range(ntrials):
            # get shown options
            opt1, opt2 = option_pairs[b, t, 0], option_pairs[b, t, 1]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # get probability of the chosen option based on sum of Qs
            c = letter_to_idx[choices[b, t]] if not isinstance(choices[b, t], (int, np.integer)) else int(choices[b, t])
            shown_vals = np.array([EV_self[b, t, o1] + EV_other[b, t, o1],
                                   EV_self[b, t, o2] + EV_other[b, t, o2]], dtype=float)
            probs_two  = softmax(shown_vals, beta)  # len=2
            ch_prob[b, t, o1] = probs_two[0]
            ch_prob[b, t, o2] = probs_two[1]
            nll += -np.log((probs_two[0] if c == o1 else probs_two[1]) + 1e-12)

            # compute prediction errors vs their respective Qs
            pe_self[b, t]  = outcomes_self[b, t]  - EV_self[b,  t, c]
            pe_other[b, t] = outcomes_other[b, t] - EV_other[b, t, c]

            pe_self_pos[b, t]  = pe_self[b, t]  if pe_self[b, t]  >= 0.0 else 0.0
            pe_self_neg[b, t]  = pe_self[b, t]  if pe_self[b, t]  <  0.0 else 0.0
            pe_other_pos[b, t] = pe_other[b, t] if pe_other[b, t] >= 0.0 else 0.0
            pe_other_neg[b, t] = pe_other[b, t] if pe_other[b, t] <  0.0 else 0.0

            # update chosen option in each Q separately
            EV_self[b,  t+1, :] = EV_self[b,  t, :].copy()
            EV_other[b, t+1, :] = EV_other[b, t, :].copy()

            EV_self[b,  t+1, c] = EV_self[b,  t, c] + (a_self_pos  * pe_self_pos[b, t] +
                                                       a_self_neg  * pe_self_neg[b, t])
            EV_other[b, t+1, c] = EV_other[b, t, c] + (a_other_pos * pe_other_pos[b, t] +
                                                       a_other_neg * pe_other_neg[b, t])

    if output == "all":
        return {
            "params": [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg],
            "choices": choices_arr,
            "outcomes_self": outcomes_self,
            "outcomes_other": outcomes_other,
            "option_pairs": option_pairs,
            "EV_self": EV_self,
            "EV_other": EV_other,
            "nll": nll,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
        }
    else:
        return calc_fval(nll, params, prior=prior, output=output)
