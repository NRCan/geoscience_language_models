# coding=utf-8
# Copyright (C) 2021 ServiceNow, Inc.
""" This file provides Auto models for training with the 
    HuggingFace Transformers repo.

"""

import warnings
from collections import OrderedDict

from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.utils import logging

# Add modeling imports here
from nrcan_p2.evaluation.bert_multiclass import BertForMultilabelSequenceClassification
from nrcan_p2.evaluation.distilbert_multiclass import DistilBertForMultilabelSequenceClassification

from transformers.models.auto.configuration_auto import (
    DistilBertConfig,
    BertConfig,
    replace_list_option_in_docstrings,
)

from transformers.models.auto.modeling_auto import (
    AUTO_MODEL_PRETRAINED_DOCSTRING
)

logger = logging.get_logger(__name__)

MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (BertConfig, BertForMultilabelSequenceClassification),
        (DistilBertConfig, DistilBertForMultilabelSequenceClassification),
    ]
)

class AutoModelForMultiLabelSequenceClassification:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    sequence classification head---when created with the
    :meth:`~transformers.AutoModelForMultiLabelSequenceClassification.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForMultiLabelSequenceClassification.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModelForMultiLabelSequenceClassification is designed to be instantiated "
            "using the `AutoModelForMultiLabelSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForMultiLabelSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING, use_model_types=False)
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a sequence classification head---from a
        configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use :meth:`~transformers.AutoModelForMultiLabelSequenceClassification.from_pretrained` to load
            the model weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoConfig, AutoModelForMultiLabelSequenceClassification
            >>> # Download configuration from huggingface.co and cache.
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> model = AutoModelForMultiLabelSequenceClassification.from_config(config)
        """
        if type(config) in MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING.keys()),
            )
        )


    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a sequence classification head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoConfig, AutoModelForMultiLabelSequenceClassification

            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = AutoModelForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased')

            >>> # Update configuration during loading
            >>> model = AutoModelForMultiLabelSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            >>> model = AutoModelForMultiLabelSequenceClassification.from_pretrained('./tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING.keys():
            return MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_MULTILABEL_CLASSIFICATION_MAPPING.keys()),
            )
        )
