import keras
import tensorflow as tf
import config
import keras_nlp
import math
# from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
from preprocess.strategies import color_masking
from keras_nlp.layers import RotaryEmbedding


# from model.decoder.CustomDecoder import CustomDecoder as TransformerDecoder
from model.logical_decoder.LogicDecoder import LogicDecoder as TransformerDecoder


# Small Settings
dense_dim = 512
heads = 1
embed_dim = 256

# Regular Settings
# dense_dim = config.dense_dim
# heads = config.heads
# embed_dim = config.embed_dim


temperature = 1.0
dropout = 0.0



@keras.saving.register_keras_serializable(package="ChessGPTa3", name="ChessGPTa3")
class ChessGPTa3(tf.keras.Model):
    def __init__(self):
        super().__init__(name='ChessGPTa3')
        self.m_type = 'a3'
        self.supports_masking = True
        self.positional = True
        self.dense_dim = dense_dim
        self.num_heads = heads
        self.embed_dim = embed_dim

        # Move Embeddings
        # self.embedding_layer = keras.layers.Embedding(
        #     config.vocab_size,
        #     self.embed_dim,
        #     mask_zero=True
        # )
        # self.positional_embedding = RotaryEmbedding()
        self.embedding_layer = TokenAndPositionEmbedding(
            vocabulary_size=config.vocab_size,
            sequence_length=config.seq_length,
            embedding_dim=self.embed_dim,
            mask_zero=True,
        )


        # Decoder Stack
        self.norm_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        # self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        # self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        # self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        # self.decoder_6 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        # self.decoder_7 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)
        # self.decoder_8 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=dropout)


        # Move Prediction Head
        self.move_prediction_head = keras.layers.Dense(
            config.vocab_size,
            name="move_prediction_head",
            activation="linear",
        )

        # Value Prediction Head
        self.value_prediction_head = keras.layers.Dense(
            1,
            name="value_prediction_head",
            activation="linear",
        )



    def call(self, inputs, training=False):

        # Inputs
        model_moves, opp_moves, is_white = inputs

        # Embed model moves
        model_move_embeddings = self.embedding_layer(model_moves)
        # model_move_embeddings = self.positional_embedding(model_move_embeddings)

        # Embed opponent moves
        opp_move_embeddings = self.embedding_layer(opp_moves)
        # opp_move_embeddings = self.positional_embedding(opp_move_embeddings)

        # Get causal mask for cross attention
        # causal_cross_mask = color_masking.generate_batch_of_causal_matrices(is_white)

        # Decoder Stack
        decoded_move = model_move_embeddings
        decoded_move, gate1 = self.decoder_1(
            decoded_move,
            encoder_sequence=opp_move_embeddings,
            # encoder_attention_mask=causal_cross_mask,
            use_causal_mask=True, use_casual_cross_mask=True,
            training=training
        )
        decoded_move, gate2 = self.decoder_2(
            decoded_move,
            encoder_sequence=opp_move_embeddings,
            # encoder_attention_mask=causal_cross_mask,
            use_causal_mask=True, use_casual_cross_mask=True,
            training=training
        )
        # decoded_move, gate3 = self.decoder_3(
        #     decoded_move,
        #     encoder_sequence=opp_move_embeddings,
        #     # encoder_attention_mask=causal_cross_mask,
        #     use_causal_mask=True, use_casual_cross_mask=True,
        #     training=training
        # )
        # decoded_move, gate4 = self.decoder_4(
        #     decoded_move,
        #     encoder_sequence=opp_move_embeddings,
        #     # encoder_attention_mask=causal_cross_mask,
        #     use_causal_mask=True, use_casual_cross_mask=True,
        #     training=training
        # )
        # decoded_move, gate5 = self.decoder_5(
        #     decoded_move,
        #     encoder_sequence=opp_move_embeddings,
        #     # encoder_attention_mask=causal_cross_mask,
        #     use_causal_mask=True, use_casual_cross_mask=True,
        #     training=training
        # )
        # decoded_move, gate6 = self.decoder_6(
        #     decoded_move,
        #     encoder_sequence=opp_move_embeddings,
        #     # encoder_attention_mask=causal_cross_mask,
        #     use_causal_mask=True, use_casual_cross_mask=True,
        #     training=training
        # )
        # decoded_move, gate7 = self.decoder_7(
        #     decoded_move,
        #     encoder_sequence=opp_move_embeddings,
        #     # encoder_attention_mask=causal_cross_mask,
        #     use_causal_mask=True, use_casual_cross_mask=True,
        #     training=training
        # )
        # decoded_move, gate8 = self.decoder_8(
        #     decoded_move,
        #     encoder_sequence=opp_move_embeddings,
        #     # encoder_attention_mask=causal_cross_mask,
        #     use_causal_mask=True, use_casual_cross_mask=True,
        #     training=training
        # )

        all_gates = []
        all_gates.extend(gate1)
        all_gates.extend(gate2)
        # all_gates.extend(gate3)
        # all_gates.extend(gate4)
        # all_gates.extend(gate5)
        # all_gates.extend(gate6)
        # all_gates.extend(gate7)
        # all_gates.extend(gate8)


        # gates_concat = tf.concat([
        #     gate1, gate2, gate3, gate4,
        #     gate5, gate6, gate7, gate8
        # ], axis=0)  # (n, batch_size, seq_length, K)


        # Move Prediction Head
        move_predictions = self.move_prediction_head(decoded_move)
        move_predictions = move_predictions / temperature

        # Value Prediction Head
        # value_predictions = self.value_prediction_head(decoded_move)


        return move_predictions, all_gates


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True, ignore_class=0)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_perplexity_tracker = keras_nlp.metrics.Perplexity(name="perplexity", from_logits=True, mask_token_id=0)

    rule_loss_tracker = tf.keras.metrics.Mean(name="rule_loss")
    rule_count_tracker = tf.keras.metrics.Mean(name="rule_count")



    def rulebook_joint_entropy_loss(self, gating_outputs, lambda_entropy=1.0, eps=1e-9):
        gates = tf.stack(gating_outputs, axis=0)          # (L, B, S, O)
        L     = tf.shape(gates)[0]
        T     = tf.shape(gates)[1] * tf.shape(gates)[2]
        gates = tf.reshape(gates, (L, T, -1))            # (L, T, O)

        p = tf.reduce_mean(gates, axis=1)                # (L, O)
        H_layer = -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1)
        entropy_term = tf.reduce_sum(H_layer)

        # ----------- corrected einsum -----------
        joint = tf.einsum('lto,ktp->lkop', gates, gates) / tf.cast(T, gates.dtype)
        # ----------------------------------------

        p_l = tf.expand_dims(tf.expand_dims(p, 1), -1)   # (L,1,O,1)
        p_k = tf.expand_dims(tf.expand_dims(p, 0), -2)   # (1,L,1,O)

        mi = joint * (tf.math.log(joint + eps)
                    - tf.math.log(p_l + eps)
                    - tf.math.log(p_k + eps))
        I_pair = tf.reduce_sum(mi, axis=[2, 3])
        off_diag = I_pair - tf.linalg.diag(tf.linalg.diag_part(I_pair))
        mi_term  = 0.5 * tf.reduce_sum(off_diag)

        return mi_term - lambda_entropy * entropy_term

    
    def load_balancing_loss(self, gates, alpha=0.01, epsilon=1e-8):
        # gates : list of (B, S, O) tensors with already-softmaxed probabilities
        stacked = tf.stack(gates, axis=0)          # (L, B, S, O)
        probs   = stacked                          # alias for clarity
        # Hard selections for p_o (fraction of tokens)
        hard_onehot = tf.one_hot(tf.argmax(probs, -1), tf.shape(probs)[-1])  # (L,B,S,O)
        p = tf.reduce_mean(hard_onehot, axis=[1,2])                          # (L, O)
        f = tf.reduce_mean(probs,         axis=[1,2])                        # (L, O)
        loss_per_layer = alpha * tf.cast(tf.shape(probs)[-1], tf.float32) \
                        * tf.reduce_sum(p * f, axis=-1)                     # (L,)
        return tf.reduce_sum(loss_per_layer)      # scalar
    
    def rule_losses(self, gating_outputs,
                    lambda_sharp=1.0,
                    lambda_div=1.0,
                    lambda_cov=0.5):
        """
        gating_outputs: list of N tensors, each (B, S, O)

        returns a scalar loss you can add to your task loss,
        and three diagnostics (H̅, H̄, C)
        """
        EPS = 1e-8   # numerical stability

        # ───────────────────────────────────────
        # stack → (N,B,S,O)
        g = tf.stack(gating_outputs, axis=0)

        # ① mean entropy  H̅   (encourages sharp selection per token)
        ent_per_token = -tf.reduce_sum(g * tf.math.log(g + EPS), axis=-1)   # (N,B,S)
        H_bar = tf.reduce_mean(ent_per_token)                               # scalar

        # ② entropy-of-mean  H̄   (encourages diversity across batch)
        mean_probs   = tf.reduce_mean(g, axis=[1,2])                        # (N,O)
        ent_of_mean  = -tf.reduce_sum(mean_probs * tf.math.log(mean_probs + EPS),
                                    axis=-1)                              # (N,)
        H_bar_over   = tf.reduce_mean(ent_of_mean)                          # scalar

        # ③ coverage  C   (penalise operators that are *never* used)
        # prob that operator o is *never* selected anywhere in batch
        never = tf.reduce_prod(1.0 - g, axis=[1,2])                         # (N,O)
        C = tf.reduce_mean(never)                                           # scalar

        loss = ( lambda_sharp * H_bar
            - lambda_div   * H_bar_over
            + lambda_cov   * C )

        return loss

    def rulebook_entropy_loss(self, gating_outputs, epsilon=1e-8):
        """
        Vectorized entropy loss to minimize the rulebook size.

        gating_outputs: list of N tensors of shape (batch, seq_len, n_operators)
        """
        # Stack along a new axis to form (N_layers, batch, seq_len, n_operators)
        stacked_gating = tf.stack(gating_outputs, axis=0)  # (N_layers, batch, seq_len, n_operators)

        # Compute mean over batch and seq_len simultaneously
        mean_probs = tf.reduce_mean(stacked_gating, axis=[1, 2])  # (N_layers, n_operators)

        # Compute entropy per layer
        entropy = -tf.reduce_sum(mean_probs * tf.math.log(mean_probs + epsilon), axis=-1)  # (N_layers,)

        # Sum entropies over all layers
        total_entropy_loss = tf.reduce_sum(entropy)

        return total_entropy_loss

    def ground_truth_rule_count(gating_outputs):
        """
        Computes non-differentiable unique rule count in a batch.

        gating_outputs: list of N tensors of shape (batch, seq_len, n_operators)
        """
        # Stack gating outputs (N_layers, batch, seq_len, n_operators)
        stacked_gating = tf.stack(gating_outputs, axis=0)

        # Argmax over operators, resulting in (N_layers, batch, seq_len)
        operator_choices = tf.argmax(stacked_gating, axis=-1, output_type=tf.int32)

        # Reshape to (batch * seq_len, N_layers)
        permuted_choices = tf.transpose(operator_choices, [1, 2, 0])  # (batch, seq_len, N_layers)
        flattened_choices = tf.reshape(permuted_choices, [-1, tf.shape(permuted_choices)[-1]])

        # Convert choices to string to uniquely identify rules
        serialized_rules = tf.strings.reduce_join(tf.strings.as_string(flattened_choices), axis=-1, separator=',')

        # Use tf.unique to identify unique rules
        unique_rules, _ = tf.unique(serialized_rules)

        # Count unique rules
        rule_count = tf.size(unique_rules)

        return rule_count
    
    
    def train_step(self, inputs):
        model_inputs, model_labels, cross_inputs, is_white = inputs
        m_inputs = [model_inputs, cross_inputs, is_white]
        with tf.GradientTape() as tape:
            # Forward Pass
            predictions, val_predcitions = self(m_inputs, training=True)
            buloss = self.pt_loss_fn(model_labels, predictions)

            # Rule Losses
            rule_loss = self.rulebook_joint_entropy_loss(val_predcitions)

            buloss = buloss + rule_loss

            # DISTRIBUTED TRAINING
            if config.distributed is True:
                uloss = tf.nn.compute_average_loss(buloss, global_batch_size=config.global_batch_size)
            else:
                uloss = buloss

            # Mixed Precision
            if config.mixed_precision is True:
                loss = self.optimizer.get_scaled_loss(uloss)
            else:
                loss = uloss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if config.mixed_precision is True:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.pt_loss_tracker.update_state(buloss)
        self.pt_perplexity_tracker.update_state(model_labels, predictions)

        rule_count = self.rulebook_entropy_loss(val_predcitions)
        # rule_count = 0
        self.rule_count_tracker.update_state(rule_count)
        self.rule_loss_tracker.update_state(rule_loss)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result(), "rule_count": self.rule_count_tracker.result(), "rule_loss": self.rule_loss_tracker.result()}

    def test_step(self, inputs):
        model_inputs, model_labels, cross_inputs, is_white = inputs
        m_inputs = [model_inputs, cross_inputs, is_white]
        predictions, val_predcitions = self(m_inputs, training=False)
        bloss = self.pt_loss_fn(model_labels, predictions)

        # Rule Losses
        rule_loss = self.rulebook_joint_entropy_loss(val_predcitions)

        # DISTRIBUTED TRAINING
        if config.distributed is True:
            loss = tf.nn.compute_average_loss(bloss, global_batch_size=config.global_batch_size)
        else:
            loss = bloss

        self.pt_loss_tracker.update_state(loss)
        self.pt_perplexity_tracker.update_state(model_labels, predictions)

        # rule_count = self.rule_count(val_predcitions)
        rule_count = 0
        self.rule_count_tracker.update_state(rule_count)

        self.rule_loss_tracker.update_state(rule_loss)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result(), "rule_count": self.rule_count_tracker.result(), "rule_loss": self.rule_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.pt_loss_tracker, self.pt_perplexity_tracker, self.rule_count_tracker, self.rule_loss_tracker]





if __name__ == '__main__':
    model = ChessGPTa3()

    # model_input = ['[white] d2d4 e2e4 a2a4']
    # opp_input = ['[black] d7d5 e7e5 a7a5']
    # is_white = [True]
    #
    # model_input_tensor = config.encode_tf(model_input)
    # opp_input_tensor = config.encode_tf(opp_input)
    # is_white = tf.convert_to_tensor(is_white)
    # inputs = [model_input_tensor, opp_input_tensor, is_white]
    #
    # output = model(inputs)
    # print(output)



    model_input = ['[start] e2e4 d2d4 b1b3 c3e4 e4g5 f1c4 c2c3 g1f3 h2h4 h4h5 g5e4']
    opp_input = ['[black] c7c6 d7d5 d5e4 b8d7 d7f6 g8h6 g7g6 f8g7 f6d5 f7f6']
    is_white = [True]

    model_input_tensor = config.encode_tf(model_input)
    opp_input_tensor = config.encode_tf(opp_input)
    is_white = tf.convert_to_tensor(is_white)
    inputs = [model_input_tensor, opp_input_tensor, is_white]

    print('Model Input Tensor:', model_input_tensor)
    print('Opponent Input Tensor:', opp_input_tensor)
    print('Is White:', is_white)
    print('White token id', config.white_token_id)
    print('Black token id', config.black_token_id)

    # output = model(inputs)
    # print(output)







