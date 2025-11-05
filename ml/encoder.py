"""
Match encoder for TFT ML pipeline.
Converts match data to X (features) and Y (placements) arrays.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from ml.vocabulary import TFTVocabulary


class TFTMatchEncoder:
    """Encodes TFT match data into training format."""

    def __init__(self, vocabulary: TFTVocabulary, max_units: int = 15):
        """
        Initialize encoder with vocabulary.

        Args:
            vocabulary: TFTVocabulary instance
            max_units: Maximum units per board (for padding)
        """
        self.vocab = vocabulary
        self.max_units = max_units

        # Calculate feature dimensions
        self.player_feature_dim = self._calculate_player_feature_dim()

    def _calculate_player_feature_dim(self) -> int:
        """Calculate feature dimension for one player."""
        # Level (1) + Units (15 × (unit_onehot + items_binary + tier)) + Traits
        unit_features = self.vocab.num_units + self.vocab.num_items + 1
        return 1 + (self.max_units * unit_features) + self.vocab.num_traits

    def encode_unit(self, unit: Dict[str, Any]) -> np.ndarray:
        """
        Encode a single unit.

        Args:
            unit: Dict with 'name', 'tier', 'items'

        Returns:
            Array of [unit_onehot, items_binary, tier]
        """
        # Unit one-hot
        unit_onehot = np.zeros(self.vocab.num_units, dtype=np.int32)
        unit_name = unit['name']
        if unit_name in self.vocab.unit_to_idx:
            unit_onehot[self.vocab.unit_to_idx[unit_name]] = 1

        # Items binary encoding
        items_binary = np.zeros(self.vocab.num_items, dtype=np.int32)
        for item_name in unit['items']:
            if item_name in self.vocab.item_to_idx:
                items_binary[self.vocab.item_to_idx[item_name]] = 1

        # Tier (star level)
        tier = np.array([unit['tier']], dtype=np.int32)

        return np.concatenate([unit_onehot, items_binary, tier])

    def encode_null_unit(self) -> np.ndarray:
        """
        Encode a null/padding unit (all zeros).

        Returns:
            Array of zeros with same shape as encode_unit output
        """
        unit_features = self.vocab.num_units + self.vocab.num_items + 1
        return np.zeros(unit_features, dtype=np.int32)

    def encode_traits(self, traits: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode traits as binary array.

        Args:
            traits: List of dicts with 'name' and 'tier_current'

        Returns:
            Array where each position is the tier_current for that trait (0 if not present)
        """
        traits_array = np.zeros(self.vocab.num_traits, dtype=np.int32)

        for trait in traits:
            trait_name = trait['name']
            if trait_name in self.vocab.trait_to_idx:
                idx = self.vocab.trait_to_idx[trait_name]
                traits_array[idx] = trait['tier_current']

        return traits_array

    def encode_player(self, participant: Dict[str, Any]) -> np.ndarray:
        """
        Encode a single player's board.

        Args:
            participant: Dict with 'level', 'units', 'traits'

        Returns:
            Flattened feature array for this player
        """
        features = []

        # Add level
        features.append(np.array([participant['level']], dtype=np.int32))

        # Add units (pad to max_units)
        units = participant['units']
        for i in range(self.max_units):
            if i < len(units):
                unit_features = self.encode_unit(units[i])
            else:
                unit_features = self.encode_null_unit()
            features.append(unit_features)

        # Add traits
        traits_features = self.encode_traits(participant['traits'])
        features.append(traits_features)

        return np.concatenate(features)

    def encode_match(self, match: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a complete match (8 players).

        Args:
            match: Dict with 'participants' list

        Returns:
            Tuple of (X, Y) where:
                X: Flattened feature array [8 * player_feature_dim]
                Y: Placements array [8]
        """
        participants = match['participants']

        # Ensure we have exactly 8 participants
        if len(participants) != 8:
            raise ValueError(f"Expected 8 participants, got {len(participants)}")

        # Encode each player
        player_features = []
        placements = []

        for participant in participants:
            player_features.append(self.encode_player(participant))
            placements.append(participant['placement'])

        # Flatten to 1D array
        X = np.concatenate(player_features)
        Y = np.array(placements, dtype=np.int32)

        return X, Y

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about feature dimensions."""
        return {
            'player_feature_dim': self.player_feature_dim,
            'match_feature_dim': self.player_feature_dim * 8,
            'max_units': self.max_units,
            'num_units': self.vocab.num_units,
            'num_items': self.vocab.num_items,
            'num_traits': self.vocab.num_traits
        }
