"""BPR Matrix Factorization - State-of-the-art baseline (RecSys 2024)."""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class BPRMF(nn.Module):
    """
    Bayesian Personalized Ranking Matrix Factorization.

    State-of-the-art baseline based on RecSys 2024 research:
    - Proper implementation critical (50% performance variance)
    - Optimizes for ranking (not rating prediction)

    Reference:
        "BPR: Bayesian Personalized Ranking from Implicit Feedback"
        Rendle et al., UAI 2009

        "Revisiting BPR: A Replicability Study" RecSys 2024

    Architecture:
        score = global_bias + user_bias + item_bias + dot(user_emb, item_emb)
        loss = BPR(score_positive, score_negative)

    All parameters are GLOBAL (standard federated learning baseline):
        - User embeddings: aggregated across clients
        - Item embeddings: aggregated across clients
        - Biases: aggregated across clients

    For personalized FL (future): user embeddings would stay local.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        """
        Initialize BPR Matrix Factorization model.

        Args:
            num_users: Total number of users
            num_items: Total number of items
            embedding_dim: Latent factor dimensionality
                          Typical values: 32, 64, 128, 256
                          Higher = more capacity, but more overfitting risk
            dropout: Dropout rate (0.0 = no dropout)
            use_bias: Whether to use bias terms
                     Biases help but add parameters
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        # User embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms (optional but recommended)
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following RecSys 2024 best practices.

        Critical for BPR performance:
            - Proper initialization prevents early saturation
            - Xavier/Glorot works well for embeddings
            - Small random values for biases
        """
        # Xavier initialization for embeddings
        init.xavier_uniform_(self.user_embeddings.weight)
        init.xavier_uniform_(self.item_embeddings.weight)

        # Bias initialization
        if self.use_bias:
            init.normal_(self.user_bias.weight, mean=0.0, std=0.01)
            init.normal_(self.item_bias.weight, mean=0.0, std=0.01)
            init.zeros_(self.global_bias)

    def _compute_score(self, user_ids, item_ids):
        """
        Compute score for user-item pairs.

        This is the core prediction function used for both
        positive and negative samples in BPR.

        Args:
            user_ids: User indices, shape (batch_size,) or (batch_size, num_samples)
            item_ids: Item indices, shape (batch_size,) or (batch_size, num_samples)

        Returns:
            scores: Predicted scores
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Apply dropout
        if self.dropout is not None:
            user_emb = self.dropout(user_emb)
            item_emb = self.dropout(item_emb)

        # Dot product
        if user_emb.dim() == 2:
            # Standard case: (batch_size, embedding_dim)
            interaction = torch.sum(user_emb * item_emb, dim=1)
        else:
            # Multiple samples: (batch_size, num_samples, embedding_dim)
            interaction = torch.sum(user_emb * item_emb, dim=2)

        # Add biases
        if self.use_bias:
            user_b = self.user_bias(user_ids).squeeze(-1)
            item_b = self.item_bias(item_ids).squeeze(-1)
            scores = self.global_bias + user_b + item_b + interaction
        else:
            scores = interaction

        return scores

    def forward(self, user_ids, pos_item_ids, neg_item_ids=None):
        """
        Forward pass for BPR training or prediction.

        Args:
            user_ids: User indices, shape (batch_size,)
            pos_item_ids: Positive (observed) item indices, shape (batch_size,)
            neg_item_ids: Negative (unobserved) item indices, shape (batch_size,) or (batch_size, num_neg)
                         If None, only returns positive scores (for prediction)

        Returns:
            If neg_item_ids is None:
                pos_scores: Scores for positive items only
            Else:
                (pos_scores, neg_scores): Tuple of positive and negative scores for BPR loss
        """
        # Compute positive scores
        pos_scores = self._compute_score(user_ids, pos_item_ids)

        # If no negative samples, return only positive scores (prediction mode)
        if neg_item_ids is None:
            return pos_scores

        # Compute negative scores for BPR training
        if neg_item_ids.dim() == 1:
            # Single negative per positive
            neg_scores = self._compute_score(user_ids, neg_item_ids)
        else:
            # Multiple negatives per positive: (batch_size, num_neg)
            # Expand user_ids to match
            batch_size, num_neg = neg_item_ids.shape
            user_ids_expanded = user_ids.unsqueeze(1).expand(batch_size, num_neg)
            neg_scores = self._compute_score(user_ids_expanded, neg_item_ids)

        return pos_scores, neg_scores

    def predict(self, user_ids, item_ids):
        """
        Predict scores for user-item pairs (inference mode).

        Args:
            user_ids: User indices
            item_ids: Item indices

        Returns:
            Predicted scores
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids, item_ids, neg_item_ids=None)
        return scores

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """
        Generate top-K recommendations for a user.

        This is where BPR shines - ranking items by predicted score.

        Args:
            user_id: User index (single user)
            top_k: Number of recommendations
            exclude_items: Items to exclude (already rated)

        Returns:
            top_items: Top-K item indices
            top_scores: Corresponding scores
        """
        self.eval()
        with torch.no_grad():
            # Expand user_id for all items
            user_ids = torch.LongTensor([user_id] * self.num_items)
            all_item_ids = torch.arange(self.num_items)

            # Get scores for all items
            scores = self._compute_score(user_ids, all_item_ids)

            # Exclude items
            if exclude_items is not None:
                scores[list(exclude_items)] = float('-inf')

            # Get top-K
            top_scores, top_items = torch.topk(scores, k=min(top_k, len(scores)))

        return top_items.cpu().numpy(), top_scores.cpu().numpy()

    def sample_negatives(
        self,
        user_ids,
        pos_item_ids,
        num_negatives=1,
        user_rated_items=None,
        sampling_strategy='uniform',
    ):
        """
        Sample negative items for BPR training.

        CRITICAL for BPR performance (RecSys 2024 finding):
            - Sampling strategy significantly affects results
            - Uniform sampling is simple and effective
            - Popularity-based sampling can help but adds complexity

        Args:
            user_ids: User indices, shape (batch_size,)
            pos_item_ids: Positive item indices, shape (batch_size,)
            num_negatives: Number of negative samples per positive
            user_rated_items: Dict[user_id -> Set[item_ids]] of rated items to exclude
            sampling_strategy: 'uniform' or 'popularity'

        Returns:
            neg_item_ids: Negative item indices, shape (batch_size, num_negatives) or (batch_size,)
        """
        batch_size = user_ids.shape[0]
        device = user_ids.device

        if num_negatives == 1:
            # Single negative per positive (most common)
            neg_items = []

            for user_id, pos_item in zip(user_ids.cpu().numpy(), pos_item_ids.cpu().numpy()):
                # Get items rated by this user
                rated = user_rated_items.get(int(user_id), set()) if user_rated_items else {int(pos_item)}

                # Sample until we get an unrated item
                while True:
                    if sampling_strategy == 'uniform':
                        neg_item = np.random.randint(0, self.num_items)
                    else:
                        # Placeholder for popularity-based sampling
                        neg_item = np.random.randint(0, self.num_items)

                    if neg_item not in rated:
                        neg_items.append(neg_item)
                        break

            neg_item_ids = torch.LongTensor(neg_items).to(device)

        else:
            # Multiple negatives per positive
            neg_items = []

            for user_id, pos_item in zip(user_ids.cpu().numpy(), pos_item_ids.cpu().numpy()):
                rated = user_rated_items.get(int(user_id), set()) if user_rated_items else {int(pos_item)}

                user_negs = []
                while len(user_negs) < num_negatives:
                    neg_item = np.random.randint(0, self.num_items)
                    if neg_item not in rated:
                        user_negs.append(neg_item)

                neg_items.append(user_negs)

            neg_item_ids = torch.LongTensor(neg_items).to(device)

        return neg_item_ids

    def get_embedding_weights(self):
        """
        Get embedding weights for regularization or analysis.

        Returns:
            Dictionary of embedding weights
        """
        return {
            'user_embeddings': self.user_embeddings.weight,
            'item_embeddings': self.item_embeddings.weight,
        }
