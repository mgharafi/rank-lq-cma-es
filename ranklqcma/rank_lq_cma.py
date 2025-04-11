"""
cma_transformations.py

This module implements an invariant surrogate assisted CMA-ES under strictly increasing transformations of the objective function, using a linear quadratic model as surrogate on rank values instead of fitness values. (see paper) https://doi.org/10.1145/3712255.3726606

The key functionalities include:
- Feature transformation and normalization of sample data.
- Cross-validation utilities for training and evaluating surrogate models.
- Computation of fitness-based transformations and ranking correlations.
- A custom model class `RankLQModel` extending `LQModel` to support
  transformation-based ensemble modeling with optional CMA function integration.

Major Components:
- `cross_validation_setup`: Prepares cross-validation splits for supervised learning.
- `get_transformations_sorting`: Ranks transformations based on Kendall tau correlation.
- `RankLQModel`: A subclass of `LQModel` that uses transformation-based ensemble models.

Dependencies:
- numpy
- scikit-learn
- scipy
- cma (including `cma.fitness_models` and `cma.ff`)

Author: Mohamed GHARAFI
Date: 11/04/2025
"""


from typing import Callable
import numpy as np
from cma.fitness_models import LQModel
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale
from scipy.stats import kendalltau
from cma import ff

def normalize(x):
    minx = np.min(x, axis=2).reshape(*(x.shape[:2]), 1)
    maxx = np.max(x, axis=2).reshape(*(x.shape[:2]), 1)
    denom = maxx - minx
    x = (x - minx) / denom
    return x

def custom_crossvalidation_set(size : int, test_size : int) -> list[int]:
    """
    Returns a list of indices for custom cross-validation.
    
    The function samples random indices without replacement with normal
    distribution centered at 0. The number of samples is equal to the
    test size.
    
    Parameters
    ----------
    size : int
        The total number of samples to draw from.
    test_size : int
        The number of samples to use for the test set.
    
    Returns
    -------
    list[int]
        A list of indices for the test set.
    """
    # sample random indices without replacement with normal distribution centered at 0
    return np.random.normal(0, 1, size).argsort()[:test_size]

def cross_validation_setup(U, Z, T, n_splits=10):
    """
    Prepares cross-validation splits for supervised learning.
    
    Parameters
    ----------
    U : array
        The input data.
    Z : array
        The linear quadratic feature data.
    T : array
        The transformation matrix.
    n_splits : int, optional
        The number of folds. If not specified, will use the default 10.
    
    Returns
    -------
    train_v : array
        The training data of the transformation for the cross-validation.
    test_v : array
        The test data of the transformation for the cross-validation.
    train_z : array
        The training data of the features for the cross-validation.
    test_z : array
        The test data of the features for the cross-validation.
    """
    # The total number of samples to draw from.
    train_data_size = len(Z)
    
    # If the number of samples is less than the number of folds, reduce the number of folds.
    if n_splits > train_data_size:
        n_splits = train_data_size
    
    # Adjust the size of the transformation matrix to be a multiple of the number of folds.
    new_data_size = n_splits * int(np.floor(train_data_size / n_splits))
    T = T[:, :new_data_size]
    Z = Z[:new_data_size]
    
    # Initialize lists to store the indices for the training and test sets.
    train_idx = []
    test_idx = []
    
    # Create a KFold object to split the data into training and test sets.
    kf = KFold(n_splits=n_splits)
    
    # Iterate over the folds and split the data into training and test sets.
    for train_index, test_index in kf.split(Z):
        # Append the indices to the lists.
        train_idx += [train_index]
        test_idx += [test_index]
    
    # Split the data into training and test sets based on the indices.
    train_v, test_v = T[:, train_idx], T[:, test_idx]
    train_z, test_z = Z[train_idx], Z[test_idx]
    
    return train_v, test_v, train_z, test_z

def get_transformations_sorting(train_v, test_v, train_z, test_z):
    """
    Ranks transformations based on the Kendall tau correlation.

    Parameters
    ----------
    train_v : array
        Training transformation matrix.
    test_v : array
        Testing transformation matrix.
    train_z : array
        Training feature matrix.
    test_z : array
        Testing feature matrix.

    Returns
    -------
    sorted_indices : array
        Indices of transformations sorted by correlation in descending order.
    """
    # Compute the pseudo-inverse of the training feature matrix
    train_pinv = np.linalg.pinv(train_z)
    
    # Calculate model coefficients using Einstein summation
    coefficients = np.einsum('bij,kbj->kbi', train_pinv, train_v)
    
    # Predict the transformations on the test feature matrix
    predictions = np.einsum('kbi,bmi->kbm', coefficients, test_z)
    
    # Compute the Kendall tau correlation for each prediction
    correlation = [
        [
            kendalltau(pp, tt).statistic 
            for pp, tt in zip(pred, test_vv)
        ] 
        for pred, test_vv in zip(predictions, test_v)
    ]
    
    # Average the correlation across all predictions
    correlation = np.mean(np.asarray(correlation), axis=1)
    
    # Return the indices of transformations sorted by correlation in descending order
    return np.argsort(correlation)[::-1]


def cma_filter(attr : str) -> bool:
    return (
            not attr.startswith(('__', 'grad'))
            and attr not in [
                'somenan', 'rot', 'flat', 'epslow', 'leadingones', 'normalSkew', 'BBOB', 'fun_as_arg', 
                # 'cornerelli',
                # 'cornerellirot',
                # 'cornersphere',
                # 'elliwithoneconstraint',
                # 'lincon',
                # 'lineard',
                'sphere_pos',
                # 'spherewithnconstraints',
                # 'spherewithoneconstraint',
                'binary_foffset',
                'binary_optimum_interval',
                'evaluations',
            ]
            # and isinstance(getattr(ff, attr), MethodType)
            )


def cma_problem_list() -> list[str]:
    return [attr for attr in dir(ff) if cma_filter(attr)]

def get_problem(attr : str) -> Callable:
    values : dict[tuple[float], float] = {}
    fun : Callable= getattr(ff, attr)
    def inner(x : list[float]) -> float:
        if tuple(x) not in values:
            try:
                val : float = fun(x)
            except Exception as e:
                print(attr, e)
            values[tuple(x)] = val
        return values[tuple(x)]
    return inner


CMA_PROBLEMS = list(map(get_problem, cma_problem_list()))

def cma_ff_transf(X : list[list[float]], CMA_PROBLEMS : list[Callable]) -> list[list[float]]:
    ys : list[list[float]] = []
    for fun in CMA_PROBLEMS:
        try:
            y : list[float] = [fun(x) for x in X]
        except Exception as e:  # noqa: F841
            continue
        if np.isnan(y).any():
            continue
        if np.isinf(y).any():
            continue
        y = minmax_scale(y)
        y = np.sort(y)
        ys.append(y.flatten())
    return np.asarray(ys, dtype=float)

def repeat(arr, reps): return np.tile(arr, reps).reshape(reps, -1)

def transformations(S, es, m):
    distance_to_min = np.linalg.norm(S - S[0], axis=1)
    mean = es.mean
    mahalanobis = np.asarray([es.mahalanobis_norm(x - mean) for x in S])
    D = repeat(distance_to_min, m // 2)
    M = repeat(mahalanobis, m // 2)
    T = np.sort(D, axis=1) ** np.abs(np.random.normal(0, 1, (m // 2, 1)))
    T = np.vstack((T, np.sort(M, axis=1) ** np.abs(np.random.normal(0, 1, (m // 2, 1)))))

    minT = T[:, 0].reshape(-1, 1)
    maxT = T[:, -1].reshape(-1, 1)

    T -= minT
    T /= maxT - minT

    return np.sort(distance_to_min), T

class RankLQModel(LQModel):

    def __init__(self, use_cma_transformation=False, **super_args):
        """
        Initializes the RankLQModel with optional CMA transformation.

        Parameters
        ----------
        use_cma_transformation : bool, optional
            Flag to indicate whether to use CMA transformation, by default False.
        **super_args : dict
            Additional arguments to pass to the superclass initializer.
        """
        # Initialize the superclass with provided arguments
        super().__init__(**super_args)
        
        # Store the flag for using CMA transformation
        self.use_cma_transformation = use_cma_transformation
    @property
    def coefficients(self):
        """model coefficients that are linear in self.expand(.)"""
        if self._coefficients_count < self.count:
            self._coefficients_count = self.count
            self._coefficients = self.compute_coefficients(self.pinv, self.Y)
            self.logger.push()  # use logging_trace attribute and xopt
            self.log_eigenvalues.push()
        
        return self._coefficients
    
    def compute_coefficients(self, pinv, Y):
        n = self.settings.n_for_model_building(self)
        return self.ensemble_model(pinv, Y, n)
    
    def ensemble_model(self, pinv, Y, n):

        size = self.settings.n_for_model_building(self)
        idx = self.settings.sorted_index(self)  # by default argsort(self.Y)
        
        if size < self.size:
            idx = idx[:size]

        U = self.X[idx]
        V = self.F[idx]
        V /= max(V)
        Z = self.Z[idx]
        T = np.vstack((V, np.linspace(0, 1, size)))
        
        sorted_models = get_transformations_sorting(*cross_validation_setup(U,Z,T))
        
        if not self.weighted:
            return np.linalg.pinv(Z) @ T[sorted_models[0]]
        else:
            E = T[sorted_models[0]]
            A = (self.sorted_weights(n) * np.asarray(E).T).T
            return pinv @ A