import torch.nn as nn
from blocks import *

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, heads: int = 8,
                      dropout: float = 0.1, d_ff: int = 2048):
    
    # Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Pos encoder layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Encoder blocks with size of N
    encoder_blks = []
    for _ in range(N):
        encoder_attn_blk = MultiHeadAttention(d_model, heads, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_blk = EncoderBlock(encoder_attn_blk, ff_block, dropout)
        encoder_blks.append(encoder_blk)

    decoder_blks = []
    for _ in range(N):
        decoder_self_attn_blk = MultiHeadAttention(d_model, heads, dropout)
        decoder_crs_attn_blk = MultiHeadAttention(d_model, heads, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_blk = DecoderBlock(decoder_self_attn_blk, decoder_crs_attn_blk, ff_block, dropout)
        decoder_blks.append(decoder_blk)

    encoder = Encoder(nn.ModuleList(encoder_blks))
    decoder = Decoder(nn.ModuleList(decoder_blks))

    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer