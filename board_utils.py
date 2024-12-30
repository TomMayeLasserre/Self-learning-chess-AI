import chess
import torch

board_size = 8


def square_to_coords(sq: int):
    """Convertit un index [0..63] en (row, col). row=0 = haut, row=7 = bas."""
    return (sq // board_size, sq % board_size)


def coords_to_square(row, col):
    return row * board_size + col


def on_board(row, col):
    return 0 <= row < board_size and 0 <= col < board_size


def encode_board(board: chess.Board, history: list = None) -> torch.Tensor:
    """
    Encode le board dans un tenseur [43, 8, 8].
      - 0-11 : Positions des pièces (6 canaux pour blanc, 6 pour noir)
      - 12   : Côté au jeu (1.0 si blanc, -1.0 si noir)
      - 13-16: Droits de roque
      - 17-42: Historique éventuel (2 positions précédentes), 13 canaux chacune

    Par simplicité, on ignore ici l'historique (ou on peut l'activer).
    """
    x = torch.zeros((43, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()

    # 0-11 : pièces
    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        base = 0 if piece.color == chess.WHITE else 6
        if piece.piece_type == chess.PAWN:
            channel = base + 0
        elif piece.piece_type == chess.KNIGHT:
            channel = base + 1
        elif piece.piece_type == chess.BISHOP:
            channel = base + 2
        elif piece.piece_type == chess.ROOK:
            channel = base + 3
        elif piece.piece_type == chess.QUEEN:
            channel = base + 4
        elif piece.piece_type == chess.KING:
            channel = base + 5
        else:
            continue
        x[channel, row, col] = 1.0

    # 12 : trait
    x[12, :, :] = 1.0 if board.turn == chess.WHITE else -1.0

    # 13-16 : droits de roque
    x[13, :, :] = 1.0 if board.has_kingside_castling_rights(
        chess.WHITE) else 0.0
    x[14, :, :] = 1.0 if board.has_queenside_castling_rights(
        chess.WHITE) else 0.0
    x[15, :, :] = 1.0 if board.has_kingside_castling_rights(
        chess.BLACK) else 0.0
    x[16, :, :] = 1.0 if board.has_queenside_castling_rights(
        chess.BLACK) else 0.0

    # 17-42 : Historique
    if history:
        # Limiter l'historique à 2
        for idx_h, past_board in enumerate(history[-2:]):
            base_channel = 17 + idx_h * 13
            past_piece_map = past_board.piece_map()
            for sq, pc in past_piece_map.items():
                row = 7 - (sq // 8)
                col = sq % 8
                base_p = 0 if pc.color == chess.WHITE else 6
                if pc.piece_type == chess.PAWN:
                    channel = base_channel + 0
                elif pc.piece_type == chess.KNIGHT:
                    channel = base_channel + 1
                elif pc.piece_type == chess.BISHOP:
                    channel = base_channel + 2
                elif pc.piece_type == chess.ROOK:
                    channel = base_channel + 3
                elif pc.piece_type == chess.QUEEN:
                    channel = base_channel + 4
                elif pc.piece_type == chess.KING:
                    channel = base_channel + 5
                else:
                    continue
                x[channel, row, col] = 1.0
            x[base_channel + 12, :, :] = 1.0 if past_board.turn == chess.WHITE else -1.0

    return x
