from transformers import XLNetPreTrainedModel, XLNetModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Union, List
import numpy

from torch.autograd import Variable
from transformers.modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def sequence_cross_entropy_with_logits(logits: torch.Tensor,
                                       targets: torch.Tensor,
                                       weights: torch.Tensor,
                                       average: str = "batch",
                                       label_smoothing: float = None,
                                       gamma: float = None,
                                       eps: float = 1e-8,
                                       alpha: Union[float, List[float], torch.FloatTensor] = None
                                       ) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    gamma : ``float``, optional (default = None)
        Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        ``gamma`` is, the more focus on hard examples.
    alpha : ``float`` or ``List[float]``, optional (default = None)
        Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
        used independently with ``gamma``. If a single ``float`` is provided, it
        is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
        negative respectively. If a list of ``float`` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.
    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")
    # make sure weights are float
    weights = weights.float()
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        eps = torch.tensor(eps, device=probs_flat.device)
        probs_flat = probs_flat.min(1 - eps)
        probs_flat = probs_flat.max(eps)
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1. - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor
    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):
            # pylint: disable=not-callable
            # shape : (2,)
            alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                        dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
            # pylint: disable=not-callable
            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
            # pylint: enable=not-callable
            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                             '{} provided.').format(type(alpha)))
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
        weights = weights * alpha_factor
    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights
    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
        return per_batch_loss
    

class XLNETForQuestionAnswering(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.XLNetConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import XLNetTokenizer, XLNetForQuestionAnsweringSimple
        import torch

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = XLNetForQuestionAnsweringSimple.from_pretrained('xlnet-base-cased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])

        outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]

        """

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (mems), (hidden_states), (attentions)

    
class XLNETForBooleanQuestionFR(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
#         config.hidden_dropout_prob = 0.0
#         config.attention_probs_dropout_prob = 0.0

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.num_classes = 2
        self.alphas = torch.tensor([[0.22, 0.78], [0.21, 0.79], [0.22, 0.73], [0.26, 0.74], [0.07, 0.93], [0.2, 0.98], [0.2, 0.98]]).to('cuda:3')
        classifiers = []
        self.criterion = []
        for item in range(7):
            classifiers.append(nn.Linear(config.d_model, self.num_classes))
            self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
        self.classifiers = nn.ModuleList(classifiers)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
    ):
        flat_input_ids = input_ids
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        
        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
        )

#         print(input_ids.shape)
        output = transformer_outputs[0]
#         print(output.shape)

        output = self.sequence_summary(output)
#         print(pooled_output.shape)
        logits = []
        for ind, logit in enumerate(output): 
            logit = self.classifiers[ind](output[ind])
            logits.append(logit)

        if labels is not None:
            loss = 0
            out_logits = []
            for ind, logit in enumerate(logits):
                loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
                out_logits.append(self.softmax(logit))
            outputs = (loss, torch.stack(out_logits),) + transformer_outputs[
                1:
            ]  # Keep mems, hidden states, attentions if there are in it

        return outputs  # (loss), reshaped_logits, (hidden_st

class XLNETForBooleanQuestionFB(XLNetPreTrainedModel):
    def __init__(self, config, device = 'cuda:0', drp = False ):
        super().__init__(config)
        
        if drp:
            config.dropout = 0.0
            config.summary_last_dropout = 0.0
        
        self.device = device
        self.transformer = XLNetModel(config)
        self.transformer_answer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.num_classes = 2

        self.classifier = nn.Linear(config.d_model, self.num_classes)
        self.alpha = torch.tensor([0.5, 0.5]).to(self.device)
        self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
        self.rnn = nn.LSTM(config.d_model, int(config.d_model/2), 1, bidirectional=True)
        self.l1 = nn.Linear(config.d_model, config.d_model)
        self.l2 = nn.Linear(config.d_model, config.d_model)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
    ):

        flat_input_ids = input_ids
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        
        outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
        )

        pooled_output = outputs[0]
        pooled_output, _ = self.rnn(pooled_output)
        pooled_output = torch.stack([pooled[-1] for pooled in pooled_output])
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.criterion(logits, labels)
            out_logits = self.softmax(logits)

            
            outputs = (loss, out_logits,)  + outputs[
                1:
            ]  # Keep mems, hidden states, attentions if there are in it

        return outputs  # (loss), reshaped_logits, (hidden_st    
    
class XLNETForBooleanQuestionFB1(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
#         config.dropout = 0.0
#         config.summary_last_dropout = 0.0
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.num_classes = 2
        self.alphas = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]]).to('cuda:7')

        classifiers = []
        self.criterion = []
        for item in range(3):
            classifiers.append(nn.Linear(config.d_model, self.num_classes))
            self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
        self.classifiers = nn.ModuleList(classifiers)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
    ):
        flat_input_ids = input_ids
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        
        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
        )

        output = transformer_outputs[0]
#         print(output.shape)
        output = self.sequence_summary(output)
#         print(output.shape)
        logits = []
        for ind, logit in enumerate(output): 
            logit = self.classifiers[ind](output[ind])
            logits.append(logit)

        if labels is not None:
            loss = 0
            out_logits = []
            for ind, logit in enumerate(logits):
                loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
                out_logits.append(self.softmax(logit))
            outputs = (loss, torch.stack(out_logits),) + transformer_outputs[
                1:
            ]  # Keep mems, hidden states, attentions if there are in it

        return outputs  # (loss), reshaped_logits, (hidden_st


class XLNETForBooleanQuestionYN(XLNetPreTrainedModel):
    def __init__(self, config, device = 'cuda:0', drp = False):
        super().__init__(config)
        
        if drp:
            config.dropout = 0.0
            config.summary_last_dropout = 0.0
    
        self.device = device
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.num_classes = 2
        self.alphas = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.27, 0.73] ]).to(self.device)
        classifiers = []
        self.criterion = []
        for item in range(3):
            classifiers.append(nn.Linear(config.d_model, self.num_classes))
            self.criterion.append(FocalLoss(alpha=self.alphas[item], class_num=self.num_classes, gamma = 2))
        self.classifiers = nn.ModuleList(classifiers)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
    ):

        flat_input_ids = input_ids
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        
        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
        )

        output = transformer_outputs[0]

        output = self.sequence_summary(output)
#         print(pooled_output.shape)
        logits = []
        for ind in range(3): 
            logit = self.classifiers[ind](output)
            logits.append(logit.squeeze(0))


        if labels is not None:
            loss = 0
            out_logits = []
            for ind, logit in enumerate(logits):
                loss += self.criterion[ind](logit.unsqueeze(0), labels[ind].unsqueeze(0))
                out_logits.append(self.softmax(logit))
            outputs = (loss, torch.stack(out_logits),) + transformer_outputs[
                1:
            ]  # Keep mems, hidden states, attentions if there are in it

        return outputs  # (loss), reshaped_logits, (hidden_st
    


class XLNETForBooleanQuestionCO(XLNetPreTrainedModel):
    def __init__(self, config, device = 'cuda:0', drp = False ):
        super().__init__(config)
        
        if drp:
            config.dropout = 0.0
            config.summary_last_dropout = 0.0
        
        self.device = device
        self.transformer = XLNetModel(config)
        self.transformer_answer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.num_classes = 2

        self.classifier = nn.Linear(config.d_model, self.num_classes)
        self.alpha = torch.tensor([0.35, 0.65]).to(self.device)
        self.criterion = FocalLoss(alpha=self.alpha, class_num=self.num_classes, gamma = 2)
        self.rnn = nn.LSTM(config.d_model, int(config.d_model/2), 1, bidirectional=True)
        self.l1 = nn.Linear(config.d_model, config.d_model)
        self.l2 = nn.Linear(config.d_model, config.d_model)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.init_weights()
        

    #@add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        input_mask=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
        labels=None,
    ):

        flat_input_ids = input_ids
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.size(-1)) if input_mask is not None else None
        
        outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
        )

        pooled_output = outputs[0]
        pooled_output, _ = self.rnn(pooled_output)
        pooled_output = torch.stack([pooled[-1] for pooled in pooled_output])
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.criterion(logits, labels)
            out_logits = self.softmax(logits)

            
            outputs = (loss, out_logits,)  + outputs[
                1:
            ]  # Keep mems, hidden states, attentions if there are in it

        return outputs  # (loss), reshaped_logits, (hidden_st