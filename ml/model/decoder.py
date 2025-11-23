"""
Decoder for TFT match encodings.
Converts encoded feature vectors back to human-readable board information.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from ml.vocabulary import TFTVocabulary
from ml.model.config import Config


class TFTMatchDecoder:
    """
    Decodes TFT match encodings back to human-readable format.
    This is the reverse of TFTMatchEncoder.
    """

    def __init__(self, vocabulary: TFTVocabulary, max_units: int = 15):
        """
        Initialize decoder with vocabulary.

        Args:
            vocabulary: TFTVocabulary instance
            max_units: Maximum units per board
        """
        self.vocab = vocabulary
        self.max_units = max_units

        # Create reverse mappings (index -> name)
        self.idx_to_unit = {idx: name for name, idx in self.vocab.unit_to_idx.items()}
        self.idx_to_item = {idx: name for name, idx in self.vocab.item_to_idx.items()}
        self.idx_to_trait = {idx: name for name, idx in self.vocab.trait_to_idx.items()}

        # Calculate dimensions (same as encoder)
        self.unit_feature_size = self.vocab.num_units + self.vocab.num_items + 1
        self.player_feature_dim = 1 + (self.max_units * self.unit_feature_size) + self.vocab.num_traits

    def decode_unit(self, unit_features: np.ndarray) -> Dict[str, Any]:
        """
        Decode a single unit's features.

        Args:
            unit_features: Array of [unit_onehot, items_binary, tier]

        Returns:
            Dict with 'name', 'items', 'tier' (or None if empty slot)
        """
        # Split features
        unit_onehot = unit_features[:self.vocab.num_units]
        items_binary = unit_features[self.vocab.num_units:self.vocab.num_units + self.vocab.num_items]
        tier = int(unit_features[-1])

        # Check if unit slot is empty (all zeros or very close to zero)
        # Use small epsilon for floating-point robustness
        if unit_onehot.sum() < 0.1 and tier == 0:
            return None

        # Decode unit name
        unit_idx = np.argmax(unit_onehot)
        unit_name = self.idx_to_unit.get(unit_idx, "Unknown")

        # Decode items
        item_indices = np.where(items_binary > 0)[0]
        items = [self.idx_to_item.get(idx, "Unknown") for idx in item_indices]

        return {
            'name': unit_name,
            'items': items,
            'tier': tier
        }

    def decode_traits(self, traits_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode traits features.

        Args:
            traits_features: Array with tier_current values for each trait

        Returns:
            List of dicts with 'name' and 'tier_current'
        """
        traits = []
        for idx, tier_current in enumerate(traits_features):
            if tier_current > 0:
                trait_name = self.idx_to_trait.get(idx, "Unknown")
                traits.append({
                    'name': trait_name,
                    'tier_current': int(tier_current)
                })

        return traits

    def decode_player(self, player_features: np.ndarray) -> Dict[str, Any]:
        """
        Decode a single player's board.

        Args:
            player_features: Flattened feature array for this player

        Returns:
            Dict with 'level', 'units', 'traits'
        """
        # Extract level
        level = int(player_features[0])

        # Extract units
        units = []
        offset = 1  # Start after level
        for i in range(self.max_units):
            unit_start = offset + (i * self.unit_feature_size)
            unit_end = unit_start + self.unit_feature_size
            unit_features = player_features[unit_start:unit_end]

            unit = self.decode_unit(unit_features)
            if unit is not None:
                units.append(unit)

        # Extract traits
        traits_start = 1 + (self.max_units * self.unit_feature_size)
        traits_features = player_features[traits_start:]
        traits = self.decode_traits(traits_features)

        return {
            'level': level,
            'units': units,
            'traits': traits
        }

    def decode_match(self, match_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode a complete match (8 players).

        Args:
            match_features: Array of shape (8, player_feature_dim)

        Returns:
            List of 8 player boards
        """
        players = []
        for i in range(Config.NUM_PLAYERS):
            player_features = match_features[i]
            player = self.decode_player(player_features)
            players.append(player)

        return players


def format_unit(unit: Dict[str, Any]) -> str:
    """Format a unit for display."""
    stars = "★" * unit['tier']
    items_str = ""
    if unit['items']:
        items_str = f" [{', '.join(unit['items'])}]"
    return f"{unit['name']} {stars}{items_str}"


def format_traits(traits: List[Dict[str, Any]]) -> str:
    """Format traits for display."""
    if not traits:
        return "None"
    trait_strs = [f"{trait['name']} ({trait['tier_current']})" for trait in traits]
    return ", ".join(trait_strs)


def print_board(player: Dict[str, Any], player_num: int, placement: int = None):
    """
    Pretty print a player's board.

    Args:
        player: Player dict with 'level', 'units', 'traits'
        player_num: Player number (1-8)
        placement: Optional placement (1-8)
    """
    placement_str = f" - Placement: {placement}" if placement is not None else ""
    print(f"\n{'='*70}")
    print(f"Player {player_num}{placement_str}")
    print(f"{'='*70}")
    print(f"Level: {player['level']}")

    print(f"\nUnits ({len(player['units'])}):")
    if player['units']:
        for unit in player['units']:
            print(f"  • {format_unit(unit)}")
    else:
        print("  (No units)")

    print(f"\nTraits:")
    print(f"  {format_traits(player['traits'])}")


def print_match_comparison(
    match_features: np.ndarray,
    true_placements: np.ndarray,
    pred_placements: np.ndarray,
    decoder: TFTMatchDecoder,
    match_num: int = 1
):
    """
    Pretty print a match with predictions vs actual placements.

    Args:
        match_features: Encoded match features (8, player_feature_dim)
        true_placements: True placements (8,)
        pred_placements: Predicted placements (8,)
        decoder: TFTMatchDecoder instance
        match_num: Match number for display
    """
    print("\n" + "="*80)
    print(f"MATCH {match_num}")
    print("="*80)

    # Decode all players
    players = decoder.decode_match(match_features)

    # Print comparison summary
    print(f"\nPlacement Comparison:")
    print(f"  Predicted: {pred_placements.tolist()}")
    print(f"  Actual:    {true_placements.tolist()}")

    # Sort players by actual placement for display
    player_order = np.argsort(true_placements)

    for rank, player_idx in enumerate(player_order, 1):
        player = players[player_idx]
        true_place = int(true_placements[player_idx])
        pred_place = int(pred_placements[player_idx])

        # Add prediction indicator
        pred_indicator = "✓" if pred_place == true_place else f"✗ (predicted {pred_place})"

        print(f"\n{'─'*70}")
        print(f"Rank {rank} - Actual: {true_place}, Predicted: {pred_place} {pred_indicator}")
        print(f"{'─'*70}")
        print(f"Level: {player['level']}")

        print(f"\nUnits ({len(player['units'])}):")
        if player['units']:
            for unit in player['units']:
                print(f"  • {format_unit(unit)}")
        else:
            print("  (No units)")

        print(f"\nTraits: {format_traits(player['traits'])}")


def print_match_predictions(
    X: torch.Tensor,
    Y_relevance: torch.Tensor,
    pred_scores: torch.Tensor,
    decoder: TFTMatchDecoder,
    num_matches: int = 3
):
    """
    Print detailed predictions for multiple matches.

    Args:
        X: Features tensor (batch, 8, player_feature_dim)
        Y_relevance: True relevance scores (batch, 8)
        pred_scores: Predicted scores (batch, 8)
        decoder: TFTMatchDecoder instance
        num_matches: Number of matches to display
    """
    from ml.model.metrics import scores_to_placements, relevance_to_placements

    # Convert to placements
    pred_placements = scores_to_placements(pred_scores)
    true_placements = relevance_to_placements(Y_relevance)

    # Convert to numpy for easier handling
    X_np = X.cpu().numpy()
    true_placements_np = true_placements.cpu().numpy()
    pred_placements_np = pred_placements.cpu().numpy()

    # Print each match
    for i in range(min(num_matches, len(X))):
        print_match_comparison(
            match_features=X_np[i],
            true_placements=true_placements_np[i],
            pred_placements=pred_placements_np[i],
            decoder=decoder,
            match_num=i + 1
        )


if __name__ == "__main__":
    # Test decoder
    print("Testing TFT Match Decoder...")

    from ml.vocabulary import TFTVocabulary

    vocab = TFTVocabulary()
    decoder = TFTMatchDecoder(vocab)

    print("✓ Decoder initialized successfully")
    print(f"  Player feature dim: {decoder.player_feature_dim}")
    print(f"  Units: {len(decoder.idx_to_unit)}")
    print(f"  Items: {len(decoder.idx_to_item)}")
    print(f"  Traits: {len(decoder.idx_to_trait)}")
