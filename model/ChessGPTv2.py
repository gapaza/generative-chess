import keras
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding

from keras_nlp.layers import RotaryEmbedding

# Small Settings
# dense_dim = config.dense_dim_small
# heads = config.heads_small
# embed_dim = config.embed_dim_small


# Regular Settings
dense_dim = config.dense_dim
heads = config.heads
embed_dim = config.embed_dim


@keras.saving.register_keras_serializable(package="ChessGPTv2", name="ChessGPTv2")
class ChessGPTv2(tf.keras.Model):
    def __init__(self):
        super().__init__(name='ChessGPTv2')
        self.m_type = 'v2'
        self.supports_masking = True
        self.dense_dim = dense_dim
        self.num_heads = heads
        self.embed_dim = embed_dim
        self.positional = True

        # Move Embeddings
        self.embedding_layer = keras.layers.Embedding(
            config.vocab_size,
            self.embed_dim,
            mask_zero=True
        )
        self.color_embedding = keras.layers.Embedding(
            3,
            self.embed_dim,
            mask_zero=True
        )
        self.piece_embedding = keras.layers.Embedding(
            8,  # [start], pawn, knight, bishop, rook, queen, king
            self.embed_dim,
            mask_zero=True
        )
        self.positional_embedding = RotaryEmbedding()

        # Decoder Stack
        self.norm_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_3 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_4 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_5 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_6 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_7 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_8 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_9 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_10 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_11 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        self.decoder_12 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        # self.decoder_13 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        # self.decoder_14 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        # self.decoder_15 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)
        # self.decoder_16 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.norm_first, dropout=config.dropout)


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
        inputs, piece_seq = inputs

        # Move Embeddings
        move_embeddings = self.embedding_layer(inputs)        # UCI embedding
        move_embeddings += self.get_color_embeddings(inputs)  # Color embedding
        move_embeddings += self.piece_embedding(piece_seq)    # Piece embedding

        # Rotary Positional Encoding
        move_embeddings = self.positional_embedding(move_embeddings)

        # Decoder Stack
        decoded_move = move_embeddings
        decoded_move = self.decoder_1(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_2(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_3(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_4(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_5(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_6(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_7(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_8(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_9(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_10(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_11(decoded_move, use_causal_mask=True, training=training)
        decoded_move = self.decoder_12(decoded_move, use_causal_mask=True, training=training)
        # decoded_move = self.decoder_13(decoded_move, use_causal_mask=True, training=training)
        # decoded_move = self.decoder_14(decoded_move, use_causal_mask=True, training=training)
        # decoded_move = self.decoder_15(decoded_move, use_causal_mask=True, training=training)
        # decoded_move = self.decoder_16(decoded_move, use_causal_mask=True, training=training)


        # Move Prediction Head
        move_predictions = self.move_prediction_head(decoded_move)

        # Value Prediction Head
        value_predictions = self.value_prediction_head(decoded_move)

        return move_predictions, value_predictions

    def get_color_embeddings(self, input):
        input_tensor = input
        batch_size = tf.shape(input)[0]
        seq_len = tf.shape(input)[1]

        # Step 1: Create a boolean mask
        mask = tf.cast(input_tensor != 0, tf.int32)

        # Step 2: Create a tensor filled with 2s
        tensor_2s = tf.ones_like(input_tensor, dtype=tf.int32) * 2

        # Step 3: Create a tensor filled with 1s
        tensor_1s = tf.ones_like(input_tensor, dtype=tf.int32)

        # Step 4: Multiply both tensors by the mask
        masked_2s = tensor_2s * mask
        masked_1s = tensor_1s * mask

        # Step 5: Gather even and odd elements
        # Create an index map where even indices map to 1s and odd indices map to 2s
        even_indices = tf.range(seq_len) % 2 == 0
        color_tensor = tf.where(even_indices, masked_1s, masked_2s)

        # Step 6: Return the final tensor
        color_embeddings = self.color_embedding(color_tensor)

        return color_embeddings




    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True, ignore_class=0)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_perplexity_tracker = keras_nlp.metrics.Perplexity(name="perplexity", from_logits=True, mask_token_id=0)

    def train_step(self, inputs):
        # input_sequences, target_sequences, piece_types = inputs
        input_sequences, target_sequences, piece_types, masks = inputs
        with tf.GradientTape() as tape:
            # Forward Pass
            predictions, val_predictions = self([input_sequences, piece_types], training=True)
            uloss = self.pt_loss_fn(target_sequences, predictions, sample_weight=masks)
            if config.mixed_precision is True:
                loss = self.optimizer.get_scaled_loss(uloss)
            else:
                loss = uloss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if config.mixed_precision is True:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.pt_loss_tracker.update_state(uloss)
        self.pt_perplexity_tracker.update_state(target_sequences, predictions, sample_weight=masks)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result()}

    def test_step(self, inputs):
        # input_sequences, target_sequences, piece_types = inputs
        input_sequences, target_sequences, piece_types, masks = inputs
        predictions, val_predictions = self([input_sequences, piece_types], training=False)
        loss = self.pt_loss_fn(target_sequences, predictions, sample_weight=masks)
        self.pt_loss_tracker.update_state(loss)
        self.pt_perplexity_tracker.update_state(target_sequences, predictions, sample_weight=masks)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result()}

    @property
    def metrics(self):
        return [self.pt_loss_tracker, self.pt_perplexity_tracker]

