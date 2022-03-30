# Copyright (C) 2021 ServiceNow, Inc.
""" Extension to sklearn's MultiOutputClassifier that 
    handles imbalanced sampling  
    See original at: 
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/multioutput.py
"""


import numpy as np
import scipy.sparse as sp
from joblib import Parallel

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.utils import check_array, check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import (check_is_fitted, has_fit_parameter,
                               _check_fit_params, _deprecate_positional_args)
from sklearn.utils.multiclass import check_classification_targets
from joblib import Parallel, delayed
from sklearn.multioutput import _fit_estimator, _partial_fit_estimator

__all__ = ["MultiOutputRegressor", "MultiOutputClassifier",
           "ClassifierChain", "RegressorChain"]

class _MyMultiOutputEstimator(MetaEstimatorMixin,
                            BaseEstimator,
                            metaclass=ABCMeta):
    @abstractmethod
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally fit the model to data.
        Fit a separate model for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.
        classes : list of ndarray of shape (n_outputs,)
            Each array is unique classes for one output in str/int
            Can be obtained by via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where y is the
            target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y,
                         force_all_finite=False,
                         multi_output=True,
                         accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        first_time = not hasattr(self, 'estimators_')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else self.estimator,
                X, y[:, i],
                classes[i] if classes is not None else None,
                sample_weight, first_time) for i in range(y.shape[1]))
        return self

    def fit(self, X, y, sample_weight=None, imbalanced_sampler=None, **fit_params):
        """ Fit the model to data.
        Fit a separate model for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        imbalanced_sampler: object that implements 
            fit_resample(X,y)->(X_resampled,y_resampled), default=None
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement"
                             " a fit method")

        X, y = self._validate_data(X, y,
                                   force_all_finite=False,
                                   multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        def over_sample(X,y,imbalanced_sampler):
            if imbalanced_sampler is None:
                return X, y

            X_resampled, ycol_resampled = imbalanced_sampler.fit_resample(X,y)
            return X_resampled, ycol_resampled

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, 
                *over_sample(X,y[:,i], imbalanced_sampler),
                sample_weight,
                **fit_params_validated)
            for i in range(y.shape[1]))
        return self

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.
        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement"
                             " a predict method")

        X = check_array(X, force_all_finite=False, accept_sparse=True)

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X)
            for e in self.estimators_)

        return np.asarray(y).T

    def _more_tags(self):
        return {'multioutput_only': True}


class MultiOutputClassifier(ClassifierMixin, _MyMultiOutputEstimator):
    """Multi target classification
    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending classifiers that do not natively support
    multi-target classification

    An extension of the original class that inherits from the imbalanced 
    sampling-compatable _MyMultiOutputEstimator. 

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit`, :term:`score` and
        :term:`predict_proba`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.
        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.
        .. versionchanged:: 0.20
           `n_jobs` default changed from 1 to None
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> X, y = make_multilabel_classification(n_classes=3, random_state=0)
    >>> clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X, y)
    >>> clf.predict(X[-2:])
    array([[1, 1, 0], [1, 1, 1]])
    """
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, Y, sample_weight=None, **fit_params):
        """Fit the model to data matrix X and targets Y.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Y : array-like of shape (n_samples, n_classes)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying classifier supports sample
            weights.
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
        """
        super().fit(X, Y, sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    @property
    def predict_proba(self):
        """Probability estimates.
        Returns prediction probabilities for each class of each output.
        This method will raise a ``ValueError`` if any of the
        estimators do not have ``predict_proba``.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data
        Returns
        -------
        p : array of shape (n_samples, n_classes), or a list of n_outputs \
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            .. versionchanged:: 0.19
                This function now returns a list of arrays where the length of
                the list is ``n_outputs``, and each array is (``n_samples``,
                ``n_classes``) for that particular output.
        """
        check_is_fitted(self)
        if not all([hasattr(estimator, "predict_proba")
                    for estimator in self.estimators_]):
            raise AttributeError("The base estimator should "
                                 "implement predict_proba method")
        return self._predict_proba

    def _predict_proba(self, X):
        results = [estimator.predict_proba(X) for estimator in
                   self.estimators_]
        return results

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples, n_outputs)
            True values for X
        Returns
        -------
        scores : float
            accuracy_score of self.predict(X) versus y
        """
        check_is_fitted(self)
        n_outputs_ = len(self.estimators_)
        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi target classification but has only one")
        if y.shape[1] != n_outputs_:
            raise ValueError("The number of outputs of Y for fit {0} and"
                             " score {1} should be same".
                             format(n_outputs_, y.shape[1]))
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))

    def _more_tags(self):
        # FIXME
        return {'_skip_test': True}