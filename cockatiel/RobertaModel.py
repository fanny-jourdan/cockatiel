"""
This module implements a RoBERTa model with a non-negative (ReLU) last layer to allow us to discover
concepts using COCKATIEL.
We use the `features` method to compute the activations on which to apply the NMF, `end_model` to go
from these activations to the final prediction.
"""

import torch
from torch import nn
from torch.nn import MSELoss
from transformers import RobertaPreTrainedModel, RobertaModel


class CustomRobertaClassificationHead(nn.Module):
    """
    A custom fully-connected classification head for RoBERTa with a non-negative layer on which
    we can compute the NMF.

    Parameters
    ----------
    config
        An object indicating the hidden layer size, the presence and amount of dropout.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def features(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        return x

    def end_model(self, x):
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
    """
    A custom RoBERTa model using a custom fully-connected head with a non-negative layer on which
    we can compute the NMF.

    Parameters
    ----------
    config
        An object indicating the hidden layer size, the presence and amount of dropout for the classification head.
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = CustomRobertaClassificationHead(config)

        self.mse_loss = MSELoss()

        self.post_init()

    def features(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        features = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self.classifier.features(features[0][:, 0, :])

    def end_model(self, activations):
        return self.classifier.end_model(activations)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)

        return logits
