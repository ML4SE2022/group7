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
