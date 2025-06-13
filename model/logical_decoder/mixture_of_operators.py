from keras import ops
import keras
import tensorflow as tf
import config
import keras_nlp
import math
# from keras_nlp.layers import TransformerDecoder
from model.decoder.CustomDecoder import CustomDecoder as TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from preprocess.strategies import color_masking
from keras_nlp.layers import RotaryEmbedding

from model.logical_decoder.sparsemax import sparsemax


from keras_nlp.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_nlp.src.utils.keras_utils import clone_initializer
from keras_nlp.src.layers.modeling.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)



class MoOLayer(tf.keras.layers.Layer):
    """Mixture-of-Operators block (FOL-aware)   D = hidden width, K = #operators."""
    def __init__(self, d, K=6, num_heads=4, use_sparsemax=True, **kwargs):
        super().__init__(**kwargs)
        self.d, self.K = d, K
        self.use_sparsemax = use_sparsemax
        # self.qkv = tf.keras.layers.Dense(3 * d, use_bias=False)

        # shared linear maps for operator bank:   (K , 2d , d)
        self.op_proj = self.add_weight(
            shape=(K, 2 * d, d), initializer="glorot_uniform", name="op_proj")
        self.gate_dense = tf.keras.layers.Dense(K, use_bias=True)


    def call(self, inputs, training=None, mask=None):
        tokens, token_literals, context_literals = inputs
        # tokens:           (B, T, d)  tokens before the attention operation
        # token_literals:   (B, T, d)  tokens after the attention operation
        # context_literals: (B, T, T)  attention scores over token literals (averaged over attention heads)
        h = tokens
        m = token_literals
        att = context_literals
        li = tf.concat([h, m], axis=-1)  # literal input (B, T, 2d)

        # Base linear projections for each operator 
        op_lin = tf.einsum('btd,kdf->btkf', li, self.op_proj)  # (B, T, K, d)

        # Fuzzy truth-value channel (slot 0 of each operator embedding)
        eps = 1e-6
        p = tf.sigmoid(li[..., :1])  # (B, T, 1) ~ fuzzy truth score (0,1) for each token literal
        logp = tf.math.log(p + eps)
        p_not = 1.0 - p
        p_and = tf.exp(tf.matmul(att, logp))  # soft-product
        # p_or = 1.0 - tf.exp(tf.reduce_sum(tf.math.log(1.0 - p + eps), axis=-1, keepdims=True))
        p_or = 1.0 - tf.exp(tf.matmul(att, tf.math.log(1.0 - p + eps)))  # soft-OR via De Morgan
        p_exists = tf.matmul(att, p)  # soft-∃
        p_all = tf.exp(tf.matmul(att, logp))  # soft-∀ (Gödel t-norm ≈ min)
        p_stack = tf.concat([
            p,  # identity (no logic applied)
            p_not,  # not (flips the truth value)
            p_and,  # and (conjunction over attended token predicates)
            p_or,  # or (disjunction over attended token predicates)
            p_exists,  # exists
            p_all
        ], axis=-1)  # (B, T, K)
        p_stack = tf.expand_dims(p_stack, -1)  # (B, T, K, 1)
        op_outs = tf.concat([p_stack, op_lin[..., 1:]], axis=-1)  # (B,T,K,d)

        # Gating / mixture
        logits = self.gate_dense(li)  # (B, T, K)
        g = sparsemax(logits, axis=-1) if self.use_sparsemax else tf.nn.softmax(logits, axis=-1)
        y = tf.reduce_sum(g[..., tf.newaxis] * op_outs, axis=-2)  # convex mixture (B, T, d)

        return h + y, g  # residual add, return gate for regularisers


    # def call(self, h, training=None, mask=None):
    #     # h: (B, T, d)
    #     B, T, d = tf.unstack(tf.shape(h))
    #     qkv = self.qkv(h)  # (B, T, 3d)
    #     q, k, v = tf.split(qkv, 3, axis=-1)
    #     scale = 1.0 / math.sqrt(float(d))
    #     att = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * scale, axis=-1)  # (B, T, T)
    #     print('attention:', att.shape)
    #     m = tf.matmul(att, v)  # neighbourhood summary (B, T, d)
    #     print('neighborhood summary:', m.shape)
    #     return self.call_logic(h, m, att, training=training, mask=mask)


if __name__ == '__main__':
    import numpy as np

    # Example usage
    d = 64  # hidden dimension
    K = 6   # number of operators
    batch_size = 2
    seq_length = 10

    layer = MoOLayer(d, K)
    tokens = tf.random.uniform((batch_size, seq_length, d))
    token_literals = tf.random.uniform((batch_size, seq_length, d))
    context_literals = tf.random.uniform((batch_size, seq_length, seq_length))

    outputs, gates = layer((tokens, token_literals, context_literals))
    print("Outputs shape:", outputs.shape)  # Should be (batch_size, seq_length, d)
    print("Gates shape:", gates.shape)  # Should be (batch_size, seq_length, K)



