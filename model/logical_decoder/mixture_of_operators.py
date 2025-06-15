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
        self.supports_masking = True
        self.d, self.K = d, K
        self.use_sparsemax = use_sparsemax

        # shared linear maps for operator bank:   (K , 2d , d)
        # self.op_proj = self.add_weight(shape=(K, 2 * d, d), initializer="glorot_uniform", name="op_proj")
        self.op_proj = self.add_weight(shape=(K, 2*d, d), initializer="glorot_uniform", name="op_proj")
        self.gate_dense = tf.keras.layers.Dense(K, use_bias=True)

        self.fuzzy_truth_dense = tf.keras.layers.Dense(1, activation='sigmoid', name="fuzzy_truth_dense")
        # self.fuzzy_expansion_dense = tf.keras.layers.Dense(d, activation='sigmoid', name="fuzzy_expansion_dense")
        self.fuzzy_expansion_dense = self.add_weight(shape=(K, 1, d), initializer="glorot_uniform", name="fuzzy_expansion_dense")


    def call(self, inputs, training=None, mask=None):
        token_literals, context_literals, attention_scores = inputs
        # token_literals:     (B, T, d)  tokens before the attention operation
        # context_literals:   (B, T, d)  tokens after the attention operation
        # attention_scores:   (B, T, T)  attention scores over token literals (averaged over attention heads)
        h = token_literals
        m = context_literals
        att = attention_scores
        li = tf.concat([h, m], axis=-1)  # literal input (B, T, 2d)
        # li = h  # literal input (B, T, d)

        # Base linear projections for each operator
        op_lin = tf.einsum('btd,kdf->btkf', li, self.op_proj)  # (B, T, K, d)

        # Fuzzy truth-value channel (slot 0 of each operator embedding)
        eps = 1e-6
        # p = tf.sigmoid(li[..., :1])  # (B, T, 1) ~ fuzzy truth score (0,1) for each token literal
        p = self.fuzzy_truth_dense(li)
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
        p_stack = tf.expand_dims(p_stack, -1)  # (B, T, K, 1) a single fuzzy truth value per token per operator

        # p_stack = self.fuzzy_expansion_dense(p_stack)  # (B, T, K, d) expand to d-dimensions
        # p_stack = tf.einsum('btd,kdf->btkf', p_stack, self.fuzzy_expansion_dense)  # (B, T, K, d)
        p_stack = tf.einsum('btki,kid->btkd', p_stack, self.fuzzy_expansion_dense)

        op_outs = p_stack + op_lin
        # op_outs = tf.concat([p_stack, op_lin[..., 1:]], axis=-1)  # (B,T,K,d)

        # Gating / mixture
        logits = self.gate_dense(li)  # (B, T, K)
        # g = sparsemax(logits, axis=-1) if self.use_sparsemax else tf.nn.softmax(logits, axis=-1)
        g = tf.nn.softmax(logits, axis=-1)
        y = tf.reduce_sum(g[..., tf.newaxis] * op_outs, axis=-2)  # convex mixture (B, T, d)

        return y, g  # no residual add, return gate for regularisers
        # return h + y, g  # residual add, return gate for regularisers


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



