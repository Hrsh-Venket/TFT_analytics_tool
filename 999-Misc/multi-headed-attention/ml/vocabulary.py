"""
Vocabulary builder for TFT ML pipeline.
Loads unit/item/trait names from CSV mappings.
"""

import csv
from typing import Dict, List


def load_vocabulary_from_csv(csv_path: str) -> List[str]:
    """
    Load vocabulary (new_name column) from CSV mapping file.

    Args:
        csv_path: Path to CSV file with old_name,new_name columns

    Returns:
        List of names (sorted alphabetically for consistency)
    """
    names = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row['new_name'])
    return sorted(names)


def create_name_to_index_mapping(names: List[str]) -> Dict[str, int]:
    """
    Create name -> index mapping for one-hot encoding.

    Args:
        names: List of names

    Returns:
        Dictionary mapping name to index
    """
    return {name: idx for idx, name in enumerate(names)}


class TFTVocabulary:
    """Vocabulary container for units, items, and traits."""

    def __init__(self, mappings_dir: str = 'name_mappings'):
        """
        Load all vocabularies from CSV files.

        Args:
            mappings_dir: Directory containing CSV mapping files
        """
        # Load names from CSVs
        self.unit_names = load_vocabulary_from_csv(f'{mappings_dir}/units_mapping.csv')
        self.item_names = load_vocabulary_from_csv(f'{mappings_dir}/items_mapping.csv')
        self.trait_names = load_vocabulary_from_csv(f'{mappings_dir}/traits_mapping.csv')

        # Create index mappings
        self.unit_to_idx = create_name_to_index_mapping(self.unit_names)
        self.item_to_idx = create_name_to_index_mapping(self.item_names)
        self.trait_to_idx = create_name_to_index_mapping(self.trait_names)

    @property
    def num_units(self) -> int:
        return len(self.unit_names)

    @property
    def num_items(self) -> int:
        return len(self.item_names)

    @property
    def num_traits(self) -> int:
        return len(self.trait_names)

    def get_stats(self) -> Dict[str, int]:
        """Get vocabulary sizes."""
        return {
            'units': self.num_units,
            'items': self.num_items,
            'traits': self.num_traits
        }
