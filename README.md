# Self-learning-chess-AI
A neural network that learns to play chess from scratch (RL) by playing against itself, inspired by AlphaZero.

## Learning Process
1. **Self-Play**:  
   - The AI plays against itself, generating games where each position and its outcome (win, lose, draw) are stored.  
   - Moves are guided by a **Monte Carlo Tree Search (MCTS)**, which uses the network’s predictions to explore promising moves.

   - **UCB Formula in MCTS**:  
     To balance exploration and exploitation, MCTS uses the Upper Confidence Bound (UCB) formula to evaluate child nodes:
     \[
     UCB = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}
     \]
     where:
     - \(Q(s, a)\): Average value of action \(a\) from state \(s\).  
     - \(P(s, a)\): Prior probability of action \(a\) from the network.  
     - \(N(s)\): Total visit count of the parent node.  
     - \(N(s, a)\): Visit count of the child node for action \(a\).  
     - \(c_{puct}\): A constant controlling exploration.

2. **Training Targets**:  
   - **Policy target \(\pi(a|s)\)**: The improved move probabilities from MCTS.  
   - **Value target \(z\)**: The game outcome (\(+1\), \(0\), \(-1\) for win, draw, lose).

3. **Loss Function**:  
   The loss combines two objectives:
   - **Policy Loss**: Cross-entropy between MCTS probabilities \(\pi(a|s)\) and network predictions \(\hat{\pi}(a|s)\):
     \[
     \mathcal{L}_{\text{policy}} = -\sum_{a} \pi(a|s) \log \hat{\pi}(a|s)
     \]
   - **Value Loss**: Mean squared error between the predicted value \(\hat{v}(s)\) and the actual outcome \(z\):
     \[
     \mathcal{L}_{\text{value}} = \left( z - \hat{v}(s) \right)^2
     \]
   - **Total Loss**:
     \[
     \mathcal{L} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}
     \]
