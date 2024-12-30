import numpy as np
import chess
import torch
import torch.nn as nn
import math
from move_tables import *
from board_utils import *


class MCTSNode:
    def __init__(self, board: chess.Board, parent, prior, history=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_terminal = board.is_game_over()

        # Historique
        if parent is not None:
            self.history = parent.history.copy()
        else:
            self.history = []
        self.history.append(board.copy())

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct=1.0):
        return self.q_value() + c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)


def mcts_search(root: MCTSNode, net: nn.Module, simulations=100, c_puct=1.0, device='cpu', move_to_index=None, num_moves=8):
    net = net.to(device)
    net.eval()

    for _ in range(simulations):
        node = root
        path = [node]

        # 1) Descente
        while node.children and not node.is_terminal:
            best_score = -999999.0
            best_child = None
            for move_idx, child in node.children.items():
                score = child.ucb_score(c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
            path.append(node)

        # 2) Expansion
        if not node.is_terminal:
            encoded = encode_board(
                node.board, history=None).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, value = net(encoded)
                policy = policy_logits.exp().squeeze(0).cpu().numpy()
                value = value.item()

            # Met à zéro les coups illégaux
            legal_moves = list(node.board.legal_moves)
            move_priors = np.zeros(num_moves, dtype=np.float32)

            sum_p = 0.0
            for m in legal_moves:
                idx_1840 = move_to_idx(node.board, m, move_to_index)
                if idx_1840 >= 0:
                    move_priors[idx_1840] = policy[idx_1840]
                    sum_p += policy[idx_1840]

            if sum_p > 1e-6:
                move_priors /= sum_p
            else:
                # fallback uniforme
                for m in legal_moves:
                    idx_1840 = move_to_idx(node.board, m, move_to_index)
                    if idx_1840 >= 0:
                        move_priors[idx_1840] = 1.0 / len(legal_moves)

            # Bruit de Dirichlet au root
            if node is root:
                alpha = 0.3
                epsilon = 0.25
                legal_count = len(legal_moves)
                if legal_count > 0:
                    dirichlet_dist = np.random.dirichlet([alpha]*legal_count)
                    i_legal = 0
                    for mv in legal_moves:
                        idx_1840 = move_to_idx(node.board, mv, move_to_index)
                        if idx_1840 >= 0:
                            old_p = move_priors[idx_1840]
                            new_p = (1 - epsilon)*old_p + epsilon * \
                                dirichlet_dist[i_legal]
                            move_priors[idx_1840] = new_p
                            i_legal += 1
                    s2 = move_priors.sum()
                    if s2 > 1e-6:
                        move_priors /= s2

            # Création des enfants
            for m in legal_moves:
                idx_1840 = move_to_idx(node.board, m, move_to_index)
                if idx_1840 < 0:
                    continue
                child_board = node.board.copy()
                child_board.push(m)
                child = MCTSNode(child_board, parent=node,
                                 prior=move_priors[idx_1840])
                node.children[idx_1840] = child

            leaf_value = value
        else:
            # 3) Terminal
            if node.board.is_checkmate():
                # Le joueur au trait est maté => -1 pour lui
                leaf_value = -1.0
            else:
                # Pat ou autre => 0.0
                leaf_value = 0.0

        # 4) Backprop
        for n in path:
            n.visit_count += 1
            n.total_value += leaf_value

    # 5) Distribution finale des visites
    visits = np.zeros(num_moves, dtype=np.float32)
    for move_idx, child in root.children.items():
        visits[move_idx] = child.visit_count
    if visits.sum() > 1e-6:
        visits /= visits.sum()

    return visits
