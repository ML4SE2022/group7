import torch.nn as nn
import torch
from transformers.modeling_utils import PreTrainedModel

BertLayerNorm = torch.nn.LayerNorm


class Model(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768 * 5, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec - code_vec, nl_vec * code_vec, nl_vec + code_vec), 1))
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()
        return loss, predictions


class ModelCombined(Model):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelCombined, self).__init__(encoder, config, tokenizer, args)
        self.mlp = nn.Sequential(nn.Linear(768 * 5, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 384),
                                 nn.ReLU(),
                                 nn.Linear(384, 192),
                                 nn.ReLU(),
                                 nn.Linear(192, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.SmoothL1Loss()


class ModelAST(Model):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelAST, self).__init__(encoder, config, tokenizer, args)

    def forward(self, code_inputs, ast_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, ast_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        ast_vec = outputs[bs:2 * bs]
        nl_vec = outputs[2 * bs:]
        if return_vec:
            return code_vec, ast_vec, nl_vec

        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec - code_vec, nl_vec * code_vec, ast_vec), 1))
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()
        return loss, predictions


class ModelFC(Model):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelFC, self).__init__(encoder, config, tokenizer, args)
        self.mlp = nn.Sequential(nn.Linear(768 * 5, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 384),
                                 nn.Tanh(),
                                 nn.Linear(384, 192),
                                 nn.Tanh(),
                                 nn.Linear(192, 1),
                                 nn.Sigmoid())


class ModelAF(Model):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelAF, self).__init__(encoder, config, tokenizer, args)
        self.mlp = nn.Sequential(nn.Linear(768 * 5, 768),
                                 nn.ReLU(),  # ReLU instead
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())


class ModelLoss(Model):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelLoss, self).__init__(encoder, config, tokenizer, args)
        # from https://pytorch.org/docs/0.3.0/nn.html#torch.nn.BCEWithLogitsLoss
        self.loss_func = nn.SmoothL1Loss()


class ClassicModel(Model):
    def __init__(self, encoder, config, tokenizer, args):
        super(ClassicModel, self).__init__(encoder, config, tokenizer, args)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768 * 5, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args
