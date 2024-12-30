import chess
from board_utils import *

DIRECTIONS_QUEEN = [
    (-1, 0), (1, 0),
    (0, -1), (0, 1),
    (-1, -1), (-1, 1),
    (1, -1), (1, 1)
]

KNIGHT_OFFSETS = [
    (-2, -1), (-2, 1), (2, -1), (2, 1),
    (-1, -2), (-1, 2), (1, -2), (1, 2)
]

DIRECTIONS_KING = DIRECTIONS_QUEEN

PROMOTION_PIECES = ['q']


def create_tables():
    """
    Construit deux tables globales (index <-> move) :
      index_to_move[idx] = (from_sq, to_sq, promo_piece)
      move_to_index[(from_sq, to_sq, promo_piece)] = idx

    'promo_piece' = None pour un coup standard
                  = 'q'   pour promotion en Dame
                  = 'castle' pour roque
    """
    index_to_move = []
    move_to_index = {}

    queen_moves_count = 0
    knight_moves_count = 0
    pawn_moves_count = 0
    king_moves_count = 0
    castle_moves_count = 0

    # 1) Coups "type Dame" (Fou + Tour + Dame)
    for from_sq in range(64):
        row0, col0 = square_to_coords(from_sq)
        for drow, dcol in DIRECTIONS_QUEEN:
            for steps in range(1, 8):
                r = row0 + drow * steps
                c = col0 + dcol * steps
                if not on_board(r, c):
                    break
                to_sq = coords_to_square(r, c)
                item = (from_sq, to_sq, None)
                if item not in move_to_index:
                    index_to_move.append(item)
                    move_to_index[item] = len(index_to_move) - 1
                    queen_moves_count += 1

    # 2) Coups de Cavalier
    for from_sq in range(64):
        row0, col0 = square_to_coords(from_sq)
        for (dr, dc) in KNIGHT_OFFSETS:
            r = row0 + dr
            c = col0 + dc
            if on_board(r, c):
                to_sq = coords_to_square(r, c)
                item = (from_sq, to_sq, None)
                if item not in move_to_index:
                    index_to_move.append(item)
                    move_to_index[item] = len(index_to_move) - 1
                    knight_moves_count += 1

    # 3) Pions (pas de prise en passant, promotion en Dame uniquement)
    for from_sq in range(64):
        row0, col0 = square_to_coords(from_sq)

        # Pion blanc (row0 -> row0+1)
        for dc in [-1, 0, 1]:
            r = row0 + 1
            c = col0 + dc
            if on_board(r, c):
                to_sq = coords_to_square(r, c)
                if dc == 0:
                    # Avance simple
                    if row0 == 6:
                        # Promotion
                        for promo in PROMOTION_PIECES:
                            item = (from_sq, to_sq, promo)
                            if item not in move_to_index:
                                index_to_move.append(item)
                                move_to_index[item] = len(index_to_move) - 1
                                pawn_moves_count += 1
                    else:
                        item = (from_sq, to_sq, None)
                        if item not in move_to_index:
                            index_to_move.append(item)
                            move_to_index[item] = len(index_to_move) - 1
                            pawn_moves_count += 1
                else:
                    # Capture
                    if row0 == 6:
                        # Promotion
                        for promo in PROMOTION_PIECES:
                            item = (from_sq, to_sq, promo)
                            if item not in move_to_index:
                                index_to_move.append(item)
                                move_to_index[item] = len(index_to_move) - 1
                                pawn_moves_count += 1
                    else:
                        item = (from_sq, to_sq, None)
                        if item not in move_to_index:
                            index_to_move.append(item)
                            move_to_index[item] = len(index_to_move) - 1
                            pawn_moves_count += 1

        # Double pas (pion blanc depuis row=1)
        if row0 == 1:
            r = row0 + 2
            c = col0
            if on_board(r, c):
                to_sq = coords_to_square(r, c)
                item = (from_sq, to_sq, None)
                if item not in move_to_index:
                    index_to_move.append(item)
                    move_to_index[item] = len(index_to_move) - 1
                    pawn_moves_count += 1

        # Pion noir (row0 -> row0-1)
        for dc in [-1, 0, 1]:
            r = row0 - 1
            c = col0 + dc
            if on_board(r, c):
                to_sq = coords_to_square(r, c)
                if dc == 0:
                    # Avance simple
                    if row0 == 1:
                        # Promotion
                        for promo in PROMOTION_PIECES:
                            item = (from_sq, to_sq, promo)
                            if item not in move_to_index:
                                index_to_move.append(item)
                                move_to_index[item] = len(index_to_move) - 1
                                pawn_moves_count += 1
                    else:
                        item = (from_sq, to_sq, None)
                        if item not in move_to_index:
                            index_to_move.append(item)
                            move_to_index[item] = len(index_to_move) - 1
                            pawn_moves_count += 1
                else:
                    # Capture
                    if row0 == 1:
                        # Promotion
                        for promo in PROMOTION_PIECES:
                            item = (from_sq, to_sq, promo)
                            if item not in move_to_index:
                                index_to_move.append(item)
                                move_to_index[item] = len(index_to_move) - 1
                                pawn_moves_count += 1
                    else:
                        item = (from_sq, to_sq, None)
                        if item not in move_to_index:
                            index_to_move.append(item)
                            move_to_index[item] = len(index_to_move) - 1
                            pawn_moves_count += 1

        # Double pas (pion noir depuis row=6)
        if row0 == 6:
            r = row0 - 2
            c = col0
            if on_board(r, c):
                to_sq = coords_to_square(r, c)
                item = (from_sq, to_sq, None)
                if item not in move_to_index:
                    index_to_move.append(item)
                    move_to_index[item] = len(index_to_move) - 1
                    pawn_moves_count += 1

    # 4) Rois (8 directions, 1 pas)
    for from_sq in range(64):
        row0, col0 = square_to_coords(from_sq)
        for drow, dcol in DIRECTIONS_KING:
            r = row0 + drow
            c = col0 + dcol
            if on_board(r, c):
                to_sq = coords_to_square(r, c)
                item = (from_sq, to_sq, None)
                if item not in move_to_index:
                    index_to_move.append(item)
                    move_to_index[item] = len(index_to_move) - 1
                    king_moves_count += 1

    # 5) Roques standard (4)
    rooks = [
        (chess.E1, chess.G1),
        (chess.E1, chess.C1),
        (chess.E8, chess.G8),
        (chess.E8, chess.C8),
    ]
    for (f, t) in rooks:
        item = (f, t, 'castle')
        if item not in move_to_index:
            index_to_move.append(item)
            move_to_index[item] = len(index_to_move) - 1
            castle_moves_count += 1

    assert len(index_to_move) == len(set(index_to_move)), "Doublon détecté !"

    print(
        f"Nombre total de coups stockés = {len(index_to_move)} (attendu ~1840).")
    print("Détails :")
    print(f"  Dames (queen-like) : {queen_moves_count}")
    print(f"  Cavaliers          : {knight_moves_count}")
    print(f"  Pions (dont promos): {pawn_moves_count}")
    print(f"  Rois               : {king_moves_count}")
    print(f"  Roques             : {castle_moves_count}")

    return move_to_index, index_to_move


def move_to_idx(board: chess.Board, move: chess.Move, move_to_index) -> int:
    """
    Transforme un chess.Move en index [0..num_moves-1].
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Roque
    if board.is_castling(move):
        key = (from_sq, to_sq, 'castle')
        if key in move_to_index:
            return move_to_index[key]

    # Promotion
    if move.promotion == chess.QUEEN:
        key = (from_sq, to_sq, 'q')
        if key in move_to_index:
            return move_to_index[key]

    # Coup standard
    key = (from_sq, to_sq, None)
    if key in move_to_index:
        return move_to_index[key]

    return -1


def idx_to_move(idx: int, index_to_move, num_moves) -> chess.Move:
    """
    Transforme un index [0..num_moves-1] en chess.Move, d'après index_to_move.
    """
    if idx < 0 or idx >= num_moves:
        return chess.Move.null()

    (from_sq, to_sq, flag) = index_to_move[idx]

    if flag == 'castle':
        return chess.Move(from_sq, to_sq)
    if flag == 'q':
        return chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

    return chess.Move(from_sq, to_sq)
