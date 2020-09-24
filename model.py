
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn

class Model(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        super().__init__()
        self.config = config
        self.word_dim = config.glove_dim
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(config.char_dim, config.char_hidden, 5)
        self.char_hidden = config.char_hidden
        self.hidden = config.hidden

        self.rnn = EncoderRNN(config.char_hidden+self.word_dim, config.hidden, 1, True, True, 1-config.keep_prob, False)

        self.cq_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_1 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)
        self.self_att = BiAttention(config.hidden*2, 1-config.keep_prob)
        self.linear_2 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
            )

        self.rnn_enc = LSTMEncoder(config.hidden, config.hidden, 1, False, True, 1-config.keep_prob, False)

        # self.dec = Decoder(input_size=config.hidden,
        #                           hidden_size=config.hidden,
        #                           word_emb=self.word_emb,
        #                           trg_vocab=self.word_emb,
        #                           n_layers=1,
        #                           device=torch.device("cuda"),
        #                           dropout= 0.,
        #                           config = config,
        #                           attention=True)
        self.embed_size = 256
        self.decoder = nn.LSTMCell(config.hidden*3,
                                   config.hidden)
        self.att_projection = nn.Linear(
            2 * config.hidden, config.hidden, bias=False)  #maybe wrong
        self.combined_output_projection = nn.Linear(
            3 * config.hidden, config.hidden, bias=False)
        self.device ="cuda"
        self.dropout = nn.Dropout(0.2)
        self.target_vocab_projection = nn.Linear(  # maybe this is wrong
            config.hidden, self.word_emb.num_embeddings)
        # self.linear_sp = nn.Linear(config.hidden*2, 1)
        #
        # self.rnn_start = EncoderRNN(config.hidden+1, config.hidden, 1, False, True, 1-config.keep_prob, False)
        # self.linear_start = nn.Linear(config.hidden*2, 1)
        #
        # self.rnn_end = EncoderRNN(config.hidden*3+1, config.hidden, 1, False, True, 1-config.keep_prob, False)
        # self.linear_end = nn.Linear(config.hidden*2, 1)
        #
        # self.rnn_type = EncoderRNN(config.hidden*3+1, config.hidden, 1, False, True, 1-config.keep_prob, False)
        # self.linear_type = nn.Linear(config.hidden*2, 3)

        self.cache_S = 0

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def generate_sent_masks(self, enc_hiddens,
                            source_lengths) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(
            enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)
    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state,
               target_padded: torch.Tensor) -> torch.Tensor:
        #: dec_init_state:Tuple[torch.Tensor, torch.Tensor
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded.transpose_(0,1)
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)  #1

        # Y = self.model_embeddings.target(target_padded)  #2
        #TODO:shal i pass embedding here??
        Y = target_padded

        for Y_t in torch.split(Y, 1):  #3
            Y_t = torch.squeeze(Y_t)
            Ybar_t = torch.cat((o_prev, Y_t), dim=1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens,
                                            enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs)

        ### END YOUR CODE

        return combined_outputs

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens,question_lens, start_mapping, end_mapping, all_mapping):
        print('word_emb_size',self.word_emb.num_embeddings)
        print('idx',ques_idxs)
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()
        ques_mask = (ques_idxs > 0).float()

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)

        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        context_word = self.word_emb(context_idxs)
        ques_word = self.word_emb(ques_idxs)
        print(ques_word.size())
        print('ques_word',ques_word) #24,45,300


        context_output = torch.cat([context_word, context_ch], dim=2) # concat of word and character
        ques_output = torch.cat([ques_word, ques_ch], dim=2)

        context_output = self.rnn(context_output, context_lens)
        ques_output = self.rnn(ques_output)
        print(ques_output.size()) #24,45,160
        print('ques_output', ques_output)
        #torch.Size([24, 1579, 160]) torch.Size([24, 45, 160])

        output = self.cq_att(ques_output, context_output, context_mask) #reverse to create c2q vector

        output = self.linear_1(output)


        output_t = self.rnn_2(output, question_lens)

        output_t = self.self_att(output_t, output_t, ques_mask)

        output_t = self.linear_2(output_t)

        output = output + output_t # output: representation of evaluation, put it through LSTM encoder. Output could be the intitial state of decoder
        enc_hiddens,dec_init =  self.rnn_enc(output,question_lens)  # start with greedy decoding

        enc_masks = self.generate_sent_masks(enc_hiddens, question_lens)

        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init,
                                       ques_output)
        P = F.log_softmax(
            self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text

        ques_output_casted = ques_output.clone().detach().to(torch.int64)[:-1]
        # Compute log probability of generating true target words
        print('P',P.size())
        print('ques',ques_output_casted.size())
        target_gold_words_log_prob = torch.gather(
            P, index=ques_output_casted[1:].unsqueeze(-1),
            dim=-1).squeeze(-1) * ques_mask[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def step(self, Ybar_t: torch.Tensor,
             dec_state,
             enc_hiddens: torch.Tensor, enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor
             ):
        # -> Tuple[Tuple, torch.Tensor, torch.Tensor]
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.squeeze(
            torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2)), 2)

        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        # enc_mask makes the probability of <paded> approaching 0
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh

        alpha_t = F.softmax(e_t, dim=1)  # attention vector
        a_t = torch.squeeze(
            torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens),
            1)  # context vector
        U_t = torch.cat((a_t, dec_hidden), 1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))

        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        # self.rnns = []
        # for i in range(nlayers):
        #     if i == 0:
        #         input_size_ = input_size
        #         output_size_ = num_units
        #     else:
        #         input_size_ = num_units if not bidir else num_units * 2
        #         output_size_ = num_units
        #     self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        # self.rnns = nn.ModuleList(self.rnns)
        self.rnn = nn.LSTM(
            input_size,
            num_units,
            dropout=dropout,
            bidirectional=True)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last
        self.h_projection = nn.Linear(
            2 * num_units,num_units, bias=False)
        self.c_projection = nn.Linear(
            2 * num_units,num_units, bias=False)

        # self.reset_parameters()


    def forward(self, input, input_lengths=None):

        output = input


        lens = input_lengths.data.cpu().numpy()


        output = self.dropout(output)
        # if input_lengths is not None:
        #     output = rnn.pack_padded_sequence(output, lens, batch_first=True)
        output, (last_hidden, last_cell) = self.rnn(rnn.pack_padded_sequence(output, lens, batch_first=True))

        output = rnn.pad_packed_sequence(output, batch_first=True)[0]
        #         if output.size(1) < slen: # used for parallel
        #             padding = Variable(output.data.new(1, 1, 1).zero_())
        #             output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
        #     if self.return_last:
        #         outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
        #     else:
        #         outputs.append(output)
        # if self.concat:
        #     return torch.cat(outputs, dim=2)

        last_hidden = torch.cat((last_hidden[0, :], last_hidden[1, :]), 1)
        init_decoder_hidden = self.h_projection(last_hidden)
        last_cell = torch.cat((last_cell[0, :], last_cell[1, :]), 1)
        init_decoder_cell = self.c_projection(last_cell)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        # hidden = hidden.view(hidden.shape[0],1,hidden.shape[1],hidden.shape[2])
        # import pickle
        # pickle.dump(output, open("lstm_output.p", "wb"))
        # pickle.dump(dec_init_state, open("lstm_dec_init.p", "wb"))
        #
        # print('LSTMencoder',output,dec_init_state[0].size(),dec_init_state[1].size())
        return output,dec_init_state

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, word_emb, trg_vocab,n_layers, device, dropout,
                 config,attention=None):
        super().__init__()
        self.output_dim = trg_vocab.num_embeddings

        self.embedding = word_emb

        self.rnn = nn.LSTM(hidden_size*3, hidden_size, n_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.attn = Attention(hidden_size=hidden_size, attn_type="general") if attention else None
        self.gen = Generator(decoder_size=hidden_size, output_dim=trg_vocab.num_embeddings )
        self.dropout = nn.Dropout(dropout)
        self.min_len_sentence = config.min_len_sentence
        self.max_len_sentence = config.sent_limit
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temperature = config.temperature
        self.decode_type = config.decode_type
        # self.special_tokens_ids = [trg_vocab.stoi[t] for t in ["<EOS>", "<PAD>"]]
        self.device = device

    def decode_rnn(self, dec_input, dec_hidden, enc_out):
        if isinstance(self.rnn, nn.GRU):
            dec_output, dec_hidden = self.rnn(dec_input, dec_hidden[0])
        else:
            #when the encoder output size change to *2 dec_input = 480

            dec_output, dec_hidden = self.rnn(dec_input, tuple(dec_hidden))

        # if self.attn:

        #     dec_output, p_attn = self.attn(dec_output, enc_out)

        dec_output = self.dropout(dec_output)

        return dec_output, dec_hidden

    def forward(self, enc_out, enc_hidden, question=None):

        batch_size = enc_out.size(0)
        #[24,45,160]
        # tensor to store decoder outputs
        outputs = []
        # print(enc_out.size(-1))
        # TODO: we should have a "if bidirectional:" statement here
        # if isinstance(enc_hidden, tuple):  # meaning we have a LSTM encoder
        #     enc_hidden = tuple(
        #         (torch.cat((hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), dim=2) for hidden in enc_hidden))
        # else:  # GRU layer
        #     enc_hidden = torch.cat((enc_hidden[0:enc_hidden.size(0):2], enc_hidden[1:enc_hidden.size(0):2]), dim=2)

        enc_out = enc_out[:, -1, :].unsqueeze(1) if not self.attn else enc_out

        dec_hidden = enc_hidden #[2,1,24,80] since enc_hidden is bidirectional
        #TODO: might have a problem here, since not sure the dim of ques_output
        if question is not None:  # TRAINING with teacher
            q_emb = question
            input_feed = torch.zeros(batch_size, 1, enc_out.size(2), device=self.device)
            print('question',question.size())
            for chunk in q_emb.split(80,-1):
                for dec_input in chunk[:, :-1, :].split(1,1):
                    print('dec_in',dec_input.size())
                    print('input',input_feed.size())
                    dec_input = torch.cat((dec_input, input_feed), 2)
                    print(dec_input.size())
                    dec_output, dec_hidden = self.decode_rnn(dec_input, dec_hidden, enc_out)

                    outputs.append(self.gen(dec_output))
                    input_feed = dec_output

        else:  # EVALUATION
            if self.decode_type not in ["topk", "beam", "greedy"]:
                print("The decode_type config value needs to be either topk, beam or greedy.")
                return outputs
            if self.decode_type == "topk":
                outputs = self.top_k_top_p_decode(dec_hidden, enc_out)
            elif self.decode_type == "beam":
                outputs = self.beam_decode(dec_hidden, enc_out)
            else:
                outputs = self.greedy_decode(dec_hidden, enc_out)

        return outputs

class Generator(nn.Module):
    def   __init__(self, decoder_size, output_dim):
        super(Generator, self).__init__()
        self.gen_func = nn.LogSoftmax(dim=-1)
        self.generator = nn.Linear(decoder_size, output_dim)

    def forward(self, x):
        out = self.gen_func(self.generator(x)).squeeze(1)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size, attn_type="dot"):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(attn_type))
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(hidden_size, hidden_size, bias=False)
            self.linear_query = nn.Linear(hidden_size, hidden_size, bias=True)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=out_bias)

    def score(self, h_t, h_s):
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            hidden_size = self.hidden_size
            wq = self.linear_query(h_t.view(-1, hidden_size))
            wq = wq.view(tgt_batch, tgt_len, 1, hidden_size)
            wq = wq.expand(tgt_batch, tgt_len, src_len, hidden_size)

            uh = self.linear_context(h_s.contiguous().view(-1, hidden_size))
            uh = uh.view(src_batch, 1, src_len, hidden_size)
            uh = uh.expand(src_batch, tgt_len, src_len, hidden_size)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, hidden_size)).view(tgt_batch, tgt_len, src_len)

    def forward(self, dec_output, enc_output, enc_output_lengths=None):
        batch, source_l, hidden_size = enc_output.size()
        batch_, target_l, hidden_size_ = dec_output.size()

        # compute attention scores, as in Luong et al.
        align = self.score(dec_output, enc_output)

        # Softmax to normalize attention weights
        align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, enc_output)

        # concatenate
        concat_c = torch.cat((c, dec_output), 2).view(batch*target_l, hidden_size*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, hidden_size)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h.permute(1, 0, 2), align_vectors
