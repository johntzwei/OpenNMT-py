"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules import Dropout
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence

from transformers import BertModel

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class BertEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, bert_str, fine_tune_bert, decoder_rnn_size,
            word_dropout=None):
        super(BertEncoder, self).__init__()

        self.fine_tune_bert = fine_tune_bert
        self.bert = BertModel.from_pretrained(bert_str)

        self.decoder_rnn_size = decoder_rnn_size

        self.word_dropout = None
        if word_dropout > 0.:
            self.word_dropout = Dropout(word_dropout)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(opt.bert,
                opt.fine_tune_bert,
                opt.dec_rnn_size,
                opt.word_dropout)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # word dropout mask
        if self.word_dropout:
            src_padding_idx = 5     # i checked 5 <-> [MASK] for bert-german-cased
            dropout_mask = self.word_dropout(torch.ones(src.shape, device=src.device)).long()
            inv_dropout_mask = 1 - dropout_mask
            src = src * dropout_mask + inv_dropout_mask * src_padding_idx

        # s_len, batch, emb_dim = emb.size()
        batch_size = src.shape[1]
        attention_mask = pad_sequence([torch.ones(l.item(), dtype=torch.long, device=src.device) for l in lengths], batch_first=True)
        
        # bert accepts [batch_size, seq_len] where each element is a token id
        bert_src = src.squeeze(2).permute([1,0])
        if self.fine_tune_bert:
            memory_bank = self.bert(bert_src, attention_mask=attention_mask)[0]    # [batch_size, seq_len, 768]
        else:
            with torch.no_grad():
                self.bert.eval()
                memory_bank = self.bert(bert_src, attention_mask=attention_mask)[0]    # [batch_size, seq_len, 768]

        # bert outputs [batch_size, seq_len, bert_dims] so it needs to be permuted
        memory_bank = memory_bank.permute([1, 0, 2])

        # only works for one-layer lstm currently
        zeros = torch.ones(1, batch_size, self.decoder_rnn_size).to(src.device)
        encoder_final = (zeros, zeros)

        return encoder_final, memory_bank, lengths

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
