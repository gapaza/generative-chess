import chess
from tqdm import tqdm

non_move_tokens = ["[pos]", "[mask]", "[white]", "[black]", "[draw]", '']


def process_puzzle_batch(inputs):
    puzzles, seq_len, worker_id = inputs
    input_sequences = []
    label_sequences = []
    piece_encodings = []
    masks = []
    if worker_id == 0:
        for puzzle in tqdm(puzzles):
            input_sequence, label_sequence, piece_encoding, mask = process_puzzle(puzzle, seq_len)
            input_sequences.append(input_sequence)
            label_sequences.append(label_sequence)
            piece_encodings.append(piece_encoding)
            masks.append(mask)
    else:
        for puzzle in puzzles:
            input_sequence, label_sequence, piece_encoding, mask = process_puzzle(puzzle, seq_len)
            input_sequences.append(input_sequence)
            label_sequences.append(label_sequence)
            piece_encodings.append(piece_encoding)
            masks.append(mask)

    return [input_sequences, label_sequences, piece_encodings, masks]


def process_puzzle(puzzle, seq_len):
    board = chess.Board()

    moves = puzzle['moves']
    moves = moves.split(' ')
    for move in moves:
        board.push_uci(move)

    line = puzzle['line']
    line_moves = line.split(' ')
    white_turn = (board.turn == chess.WHITE)
    for idx, move in enumerate(line_moves):
        white_turn = (board.turn == chess.WHITE)
        board.push_uci(move)

    # Create Input Sequence
    input_sequence = ['[start]'] + moves + line_moves
    input_sequence = ' '.join(input_sequence)

    # Create Label Sequence
    label_sequence = moves + line_moves
    if board.is_checkmate():
        if white_turn is True:  # White just mated black
            label_sequence += ['[white]']
        else:  # Black just mated white
            label_sequence += ['[black]']
    label_sequence = ' '.join(label_sequence)

    # Create Mask
    mask = [0 for _ in range(len(moves))]
    mask += [1 for _ in range(len(line_moves))]
    if board.is_checkmate():
        mask += [1]  # Predict the color that just won
    while len(mask) < seq_len:
        mask.append(0)
    if len(mask) > seq_len:
        mask = mask[:seq_len]

    # Get game piece encoding
    all_moves = ' '.join(moves + line_moves)
    piece_encoding = get_piece_encoding_padded(all_moves, seq_len)

    return input_sequence, label_sequence, piece_encoding, mask


def get_piece_encoding_padded(uci_string, seq_len):
    game_piece_encoding = get_piece_encoding(uci_string).split(' ')
    game_piece_encoding = [int(x) for x in game_piece_encoding]
    game_piece_encoding = [1] + game_piece_encoding
    while len(game_piece_encoding) < seq_len:
        game_piece_encoding.append(0)
    if len(game_piece_encoding) > seq_len:
        game_piece_encoding = game_piece_encoding[:seq_len]
    return game_piece_encoding


def get_piece_encoding(uci_string):
    uci_game_moves = uci_string.split(' ')
    game = chess.Board()
    # 0 is padding token (every position that isn't a valid uci move)

    game_piece_types = []
    for move_uci in uci_game_moves:
        try:
            if move_uci == '[start]':
                piece_type = 1
                game_piece_types.append(piece_type)
                continue
            if move_uci in non_move_tokens:
                piece_type = 0
                game_piece_types.append(piece_type)
                continue
            move = chess.Move.from_uci(move_uci)
            piece = game.piece_at(move.from_square)
            if piece is not None:
                piece_type = piece.piece_type + 1
            else:
                piece_type = 0

            game_piece_types.append(piece_type)
            game.push(move)  # Make the move on the board to update the board state
        except:
            game_piece_types.append(0)

    # No need for padding  in this function
    game_piece_types = ' '.join([str(piece_type) for piece_type in game_piece_types])
    return game_piece_types



