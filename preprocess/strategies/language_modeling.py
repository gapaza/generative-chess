import tensorflow as tf
import config
import chess




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



def preprocess_decoder_batch_piece(moves, pieces):
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
    pieces = tf.cast(pieces, tf.int16)

    return encoded_inputs, encoded_labels, pieces




