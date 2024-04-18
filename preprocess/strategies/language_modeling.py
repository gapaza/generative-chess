import tensorflow as tf
import config
import chess




def preprocess_decoder_batch(moves):
    encoded_moves = config.encode_tf(moves)

    # Inputs
    encoded_shape = tf.shape(encoded_moves)
    batch_size = encoded_shape[0]
    seq_length = encoded_shape[1]
    start_token = tf.fill([batch_size, 1], config.start_token_id)
    encoded_inputs = tf.concat([start_token, encoded_moves], axis=1)
    encoded_inputs = encoded_inputs[:, :seq_length]

    # Labels
    encoded_labels = encoded_moves

    # Optionally cast to int16 to save memory
    encoded_inputs = tf.cast(encoded_inputs, tf.int16)
    encoded_labels = tf.cast(encoded_labels, tf.int16)

    return encoded_inputs, encoded_labels



# -------------------------
# Piece Encoding
# -------------------------

def get_game_piece_encoding_from_str(uci_string):
    uci_game_moves = uci_string.split(' ')
    game = chess.Board()
    # 0 is padding token (every position that isn't a valid uci move)

    game_piece_types = []
    for move_uci in uci_game_moves:
        try:
            if move_uci in ['', '[start]'] or move_uci in config.end_of_game_tokens or move_uci in config.special_tokens:
                game_piece_types.append(0)
                continue
            move = chess.Move.from_uci(move_uci)
            piece = game.piece_at(move.from_square)
            if piece is not None:
                piece_type = piece.piece_type
            else:
                piece_type = 0

            game_piece_types.append(piece_type)
            game.push(move)  # Make the move on the board to update the board state
        except:
            game_piece_types.append(0)

    # No need for padding  in this function
    game_piece_types = ' '.join([str(piece_type) for piece_type in game_piece_types])
    return game_piece_types

def get_game_piece_encoding(text_tensor):
    uci_game = text_tensor.numpy().decode('utf-8')
    uci_game_moves = uci_game.split(' ')
    game = chess.Board()
    # print('UCI Game moves:', uci_game_moves)

    game_piece_types = []
    for move_uci in uci_game_moves:
        try:
            if move_uci in ['', '[start]'] or move_uci in config.end_of_game_tokens:
                game_piece_types.append(0)
                continue
            move = chess.Move.from_uci(move_uci)
            piece = game.piece_at(move.from_square)
            if piece is not None:
                piece_type = piece.piece_type
            else:
                piece_type = 0

            game_piece_types.append(piece_type)
            game.push(move)  # Make the move on the board to update the board state
        except:
            game_piece_types.append(0)

    # Preprocess game piece types for dataset
    # 1. Add 0 for start token
    # 2. Pad to seq_length or clip to seq_length
    game_piece_types = [0] + game_piece_types
    if len(game_piece_types) < config.seq_length:
        game_piece_types = game_piece_types + [0] * (config.seq_length - len(game_piece_types))
    if len(game_piece_types) > config.seq_length:
        game_piece_types = game_piece_types[:config.seq_length]


    return game_piece_types


def pad_piece_dataset(piece_vector):

    # Add start token to the beginning of the piece vector
    pieces_split = tf.strings.split(piece_vector, ' ')
    pieces_split = tf.strings.to_number(pieces_split, out_type=tf.int32)
    start_token = tf.fill([1], 0)
    start_token = tf.cast(start_token, tf.int32)
    pieces_split = tf.concat([start_token, pieces_split], axis=0)
    pieces_split = pieces_split[:config.seq_length]

    # Pad to seq_length
    seq_len = tf.shape(pieces_split)[0]
    padding_size = config.seq_length - seq_len
    paddings = [[0, padding_size]]  # (before1, after1), (before2, after2)
    padded_tensor = tf.pad(pieces_split, paddings, "CONSTANT")

    return padded_tensor


def preprocess_decoder_batch_piece(moves, pieces):

    # 1. Encode moves
    encoded_moves = config.encode_tf(moves)
    print('Encoded Moves', encoded_moves)

    # 2. Create move inputs
    encoded_shape = tf.shape(encoded_moves)
    batch_size = encoded_shape[0]
    seq_length = encoded_shape[1]
    start_token = tf.fill([batch_size, 1], config.start_token_id)
    encoded_inputs = tf.concat([start_token, encoded_moves], axis=1)
    encoded_inputs = encoded_inputs[:, :seq_length]

    # 3. Create move labels
    encoded_labels = encoded_moves

    # Optionally cast to int16 to save memory
    encoded_inputs = tf.cast(encoded_inputs, tf.int16)
    encoded_labels = tf.cast(encoded_labels, tf.int16)
    pieces = tf.cast(pieces, tf.int16)

    return encoded_inputs, encoded_labels, pieces




