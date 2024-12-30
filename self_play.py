from tqdm import tqdm
from mcts import *


def play_one_game(net, device='cpu', max_moves=150, mcts_sims=400,
                  c_puct=1.0, temperature_moves=30, index_to_move=None, move_to_index=None, num_moves=8):
    board = chess.Board()
    data = []
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        root = MCTSNode(board.copy(), parent=None, prior=1.0)
        pi = mcts_search(root, net, simulations=mcts_sims, c_puct=c_puct,
                         device=device, move_to_index=move_to_index, num_moves=num_moves)

        # On stocke la position encodée + la politique
        s_enc = encode_board(board)
        data.append([s_enc, pi, None])  # z (valeur finale) sera complété après

        # Choix du coup suivant
        if move_count < temperature_moves:
            # tirage au sort pondéré par pi
            move_idx = np.random.choice(len(pi), p=pi)
        else:
            # déterministe : argmax
            move_idx = np.argmax(pi)

        move = idx_to_move(move_idx, index_to_move, num_moves)
        if move not in board.legal_moves:
            # fallback : coup aléatoire légal
            move = np.random.choice(list(board.legal_moves))

        board.push(move)
        move_count += 1

    # Détermination du résultat final
    # 1 = victoire blanc, -1 = victoire noir, 0 = nullle
    if board.is_checkmate():
        # si board.turn == White => c'est au trait de jouer => celui qui vient de jouer est noir
        winner = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
        final_value = 1.0 if winner == chess.WHITE else -1.0
    else:
        final_value = 0.0

    # On affecte le même 'z' à toutes les positions de la partie
    for item in data:
        item[2] = final_value

    return data


def play_game_against(net_white, net_black, device='cpu',
                      max_moves=150, mcts_sims=400, c_puct=1.0, index_to_move=None, move_to_index=None, num_moves=8):
    board = chess.Board()
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        if board.turn == chess.WHITE:
            root = MCTSNode(board.copy(), None, 1.0)
            pi = mcts_search(root, net_white, mcts_sims, c_puct,
                             device, move_to_index, num_moves)
        else:
            root = MCTSNode(board.copy(), None, 1.0)
            pi = mcts_search(root, net_black, mcts_sims, c_puct,
                             device, move_to_index, num_moves)

        move_idx = np.argmax(pi)
        move = idx_to_move(move_idx, index_to_move, num_moves)
        if move not in board.legal_moves:
            move = np.random.choice(list(board.legal_moves))

        board.push(move)
        move_count += 1

    # Résultat du point de vue des Blancs
    if board.is_checkmate():
        # Si le board.turn est white => ce sont les blancs qui doivent jouer => donc ils viennent de se faire mater
        return -1 if board.turn == chess.WHITE else 1
    else:
        return 0  # nulle


def play_match(netA, netB, nb_games=10, device='cpu',
               max_moves=150, mcts_sims=400, c_puct=1.0, index_to_move=None,
               move_to_index=None,
               num_moves=None):
    winsA = 0
    winsB = 0
    draws = 0

    for i in tqdm(range(nb_games)):
        if i < nb_games // 2:
            # netA = Blanc
            result = play_game_against(netA, netB, device, max_moves, mcts_sims, c_puct, index_to_move,
                                       move_to_index, num_moves)
        else:
            # netB = Blanc
            result = play_game_against(netB, netA, device, max_moves, mcts_sims, c_puct, index_to_move,
                                       move_to_index, num_moves)
            # L’inversion du résultat
            result = -result

        if result == 1:
            winsA += 1
        elif result == -1:
            winsB += 1
        else:
            draws += 1

    return winsA, draws, winsB
