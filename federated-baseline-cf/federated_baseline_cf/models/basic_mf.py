"""Basic Matrix Factorization with MSE loss (Baseline)."""

import torch
import torch.nn as nn
import torch.nn.init as init


class BasicMF(nn.Module):
    """
    Basic Matrix Factorization for Collaborative Filtering.

    This is a BASELINE model for federated learning comparison.
    Uses MSE loss for rating prediction.

    Architecture:
        prediction = global_bias + user_bias + item_bias + dot(user_emb, item_emb)

    All parameters are GLOBAL (standard federated learning):
        - User embeddings: aggregated across clients
        - Item embeddings: aggregated across clients
        - Biases: aggregated across clients

    This serves as a baseline to compare against:
        1. BPR-MF (better ranking performance)
        2. Personalized FL (user embeddings local)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.0,
    ):
        """
        Initialize Basic Matrix Factorization model.

        Args:
            num_users: Total number of users in the system
            num_items: Total number of items in the catalog
            embedding_dim: Dimensionality of user/item latent factors
                          Typical values: 32, 64, 128, 256
            dropout: Dropout rate for embeddings (0.0 = no dropout)
                    Helps prevent overfitting on sparse data
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User embeddings (latent factors)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        # Item embeddings (latent factors)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # User biases (captures user rating tendencies)
        self.user_bias = nn.Embedding(num_users, 1)

        # Item biases (captures item popularity)
        self.item_bias = nn.Embedding(num_items, 1)

        # Global bias (overall rating mean)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights following best practices.

        Initialization strategy (from 2024 research):
            - Embeddings: Xavier/Glorot initialization
            - Biases: Small random values
            - Global bias: Zero (will learn mean rating)
        """
        # Xavier initialization for embeddings
        init.xavier_uniform_(self.user_embeddings.weight)
        init.xavier_uniform_(self.item_embeddings.weight)

        # Small random initialization for biases
        init.normal_(self.user_bias.weight, mean=0.0, std=0.01)
        init.normal_(self.item_bias.weight, mean=0.0, std=0.01)

        # Global bias stays at zero initially
        init.zeros_(self.global_bias)

    def forward(self, user_ids, item_ids):
        """
        Forward pass: predict ratings for user-item pairs.

        Args:
            user_ids: Tensor of user indices, shape (batch_size,)
            item_ids: Tensor of item indices, shape (batch_size,)

        Returns:
            predictions: Predicted ratings, shape (batch_size,)

        Prediction formula:
            r̂_ui = μ + b_u + b_i + q_i^T * p_u

        Where:
            μ = global_bias
            b_u = user_bias[u]
            b_i = item_bias[i]
            p_u = user_embeddings[u]
            q_i = item_embeddings[i]
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)  # (batch_size, embedding_dim)
        item_emb = self.item_embeddings(item_ids)  # (batch_size, embedding_dim)

        # Apply dropout if enabled
        if self.dropout is not None:
            user_emb = self.dropout(user_emb)
            item_emb = self.dropout(item_emb)

        # Get biases
        user_b = self.user_bias(user_ids).squeeze()  # (batch_size,)
        item_b = self.item_bias(item_ids).squeeze()  # (batch_size,)

        # Dot product of embeddings
        interaction = torch.sum(user_emb * item_emb, dim=1)  # (batch_size,)

        # Final prediction
        predictions = self.global_bias + user_b + item_b + interaction

        return predictions

    def predict(self, user_ids, item_ids):
        """
        Predict ratings (inference mode).

        Args:
            user_ids: User indices
            item_ids: Item indices

        Returns:
            Predicted ratings, clamped to valid range
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids)
            # Clamp to rating range [1, 5] for MovieLens
            predictions = torch.clamp(predictions, min=1.0, max=5.0)
        return predictions

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """
        Generate top-K recommendations for a user.

        Args:
            user_id: User index (single user)
            top_k: Number of recommendations to return
            exclude_items: Set of item indices to exclude (already rated)

        Returns:
            top_items: List of top-K item indices
            top_scores: List of corresponding scores
        """
        self.eval()
        with torch.no_grad():
            # Get user embedding
            user_ids = torch.LongTensor([user_id])
            user_emb = self.user_embeddings(user_ids)  # (1, embedding_dim)
            user_b = self.user_bias(user_ids)  # (1, 1)

            # Get all item embeddings and biases
            all_item_ids = torch.arange(self.num_items)
            item_embs = self.item_embeddings(all_item_ids)  # (num_items, embedding_dim)
            item_bs = self.item_bias(all_item_ids).squeeze()  # (num_items,)

            # Compute scores for all items
            scores = (
                self.global_bias
                + user_b.squeeze()
                + item_bs
                + torch.matmul(item_embs, user_emb.T).squeeze()
            )

            # Exclude already rated items
            if exclude_items is not None:
                scores[list(exclude_items)] = float('-inf')

            # Get top-K
            top_scores, top_items = torch.topk(scores, k=min(top_k, len(scores)))

        return top_items.cpu().numpy(), top_scores.cpu().numpy()

    def get_embedding_weights(self):
        """
        Get current embedding weights (for regularization or analysis).

        Returns:
            Dictionary of embedding weights
        """
        return {
            'user_embeddings': self.user_embeddings.weight,
            'item_embeddings': self.item_embeddings.weight,
        }
