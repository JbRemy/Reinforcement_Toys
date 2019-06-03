from .utils import ipython_info
if ipython_info() == "notebook":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing
from copy import deepcopy
from itertools import permutations, product
import numpy as np

class GridSearch(object):
    def __init__(self, scoring_function, n_jobs=1, n_estimates=1):
        if n_estimates == 1:
            self.scoring_function = scoring_function

        else:
            def scoring_function_(**kwargs):
                return np.mean([scoring_function(**kwargs) for _ in range(n_estimates)])

            self.scoring_function = scoring_function_

        self.n_jobs = n_jobs
        self.n_estimates = n_estimates
        
    def __call__(self, params_values):
        params_list = self.get_params_list_(params_values)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.scoring_function)(**{k:v for k,v in zip(params_values.keys(), params_)}) \
            for params_ in tqdm(params_list, desc="Grid search", leave=True)
        )

        best_ind = np.argmax(results)

        return {k:v for k,v in zip(params_values.keys(), params_list[best_ind])}, results[best_ind]
    
    @staticmethod
    def get_params_list_(params_values):
        out = []
        for k,l in params_values.items():
            if len(out) == 0:
                out = [[_] for _ in l]

            else:
                out = [x + [y] for x,y in product(out, l)]

        return out
