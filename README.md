# Self-learning-chess-AI
A neural network that learns to play chess from scratch (RL) by playing against itself (MCTS + neural network), inspired by AlphaZero.

## Learning Process
1. **Self-Play**:  
   - The AI plays against itself, generating games where each position and its outcome (win, lose, draw) are stored.  
   - Moves are guided by a **Monte Carlo Tree Search (MCTS)**, which uses the network’s predictions to explore promising moves.

   ### UCB Formula in MCTS  
   To balance exploration and exploitation, MCTS uses the **Upper Confidence Bound (UCB)** formula to evaluate child nodes:
UCB = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

markdown
Copier le code
where:
- `Q(s, a)`: Average value of action `a` from state `s`.  
- `P(s, a)`: Prior probability of action `a` from the network.  
- `N(s)`: Total visit count of the parent node.  
- `N(s, a)`: Visit count of the child node for action `a`.  
- `c_puct`: A constant controlling exploration.

2. **Training Targets**:  
- **Policy target**: `π(a|s)` is the improved move probabilities from MCTS.  
- **Value target**: `z` is the game outcome (`+1` for win, `0` for draw, `-1` for loss).

3. **Loss Function**:  
The loss combines two objectives:
- **Policy Loss**: Cross-entropy between MCTS probabilities `π(a|s)` and network predictions `π̂(a|s)`:
  ```
  L_policy = -Σ[π(a|s) * log(π̂(a|s))]
  ```
- **Value Loss**: Mean squared error between the predicted value `v̂(s)` and the actual outcome `z`:
  ```
  L_value = (z - v̂(s))^2
  ```
- **Total Loss**:
  ```
  L = L_policy + L_value
  ```
