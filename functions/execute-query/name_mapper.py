"""
Name Mapping Module for TFT Analytics

This module handles loading and applying name mappings for units, traits, and items
to convert raw API identifiers into clean, readable names for BigQuery storage.
"""

import csv
import logging
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

class TFTNameMapper:
    """
    Handles loading and applying name mappings for TFT data.
    
    Converts raw API identifiers (like TFT14_Jinx_34) into clean names (like Jinx).
    """
    
    def __init__(self, mappings_dir: str = "name_mappings"):
        """
        Initialize the name mapper with mapping files from the specified directory.
        
        Args:
            mappings_dir: Directory containing mapping CSV files
        """
        self.mappings_dir = mappings_dir
        self.units_mapping: Dict[str, str] = {}
        self.traits_mapping: Dict[str, str] = {}
        self.items_mapping: Dict[str, str] = {}
        
        # Load all mappings
        self._load_mappings()
    
    def _load_mappings(self):
        """Load all mapping files from the mappings directory."""
        try:
            # Load units mapping
            units_file = os.path.join(self.mappings_dir, "units_mapping.csv")
            if os.path.exists(units_file):
                self.units_mapping = self._load_mapping_file(units_file)
                logger.info(f"Loaded {len(self.units_mapping)} unit mappings")
            else:
                logger.warning(f"Units mapping file not found: {units_file}")
            
            # Load traits mapping
            traits_file = os.path.join(self.mappings_dir, "traits_mapping.csv")
            if os.path.exists(traits_file):
                self.traits_mapping = self._load_mapping_file(traits_file)
                logger.info(f"Loaded {len(self.traits_mapping)} trait mappings")
            else:
                logger.warning(f"Traits mapping file not found: {traits_file}")
            
            # Load items mapping
            items_file = os.path.join(self.mappings_dir, "items_mapping.csv")
            if os.path.exists(items_file):
                self.items_mapping = self._load_mapping_file(items_file)
                logger.info(f"Loaded {len(self.items_mapping)} item mappings")
            else:
                logger.warning(f"Items mapping file not found: {items_file}")
                
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
    
    def _load_mapping_file(self, file_path: str) -> Dict[str, str]:
        """
        Load a single mapping CSV file.
        
        Args:
            file_path: Path to the CSV mapping file
            
        Returns:
            Dictionary mapping raw names to clean names
        """
        mapping = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Expected format: raw_name,clean_name
                    if 'raw_name' in row and 'clean_name' in row:
                        raw_name = row['raw_name'].strip()
                        clean_name = row['clean_name'].strip()
                        if raw_name and clean_name:
                            mapping[raw_name] = clean_name
                    else:
                        # Try alternative column names
                        keys = list(row.keys())
                        if len(keys) >= 2:
                            raw_name = row[keys[0]].strip()
                            clean_name = row[keys[1]].strip()
                            if raw_name and clean_name:
                                mapping[raw_name] = clean_name
        except Exception as e:
            logger.error(f"Error loading mapping file {file_path}: {e}")
        
        return mapping
    
    def map_unit_name(self, raw_name: str) -> str:
        """
        Map a raw unit name to a clean name.
        
        Args:
            raw_name: Raw unit identifier from API (e.g., 'TFT14_Jinx_34')
            
        Returns:
            Clean unit name (e.g., 'Jinx') or original name if no mapping found
        """
        if not raw_name:
            return raw_name
        
        # Try exact match first
        if raw_name in self.units_mapping:
            return self.units_mapping[raw_name]
        
        # If no exact match, return original name
        # Could add fuzzy matching logic here if needed
        return raw_name
    
    def map_trait_name(self, raw_name: str) -> str:
        """
        Map a raw trait name to a clean name.
        
        Args:
            raw_name: Raw trait identifier from API (e.g., 'TFT14_Vanguard')
            
        Returns:
            Clean trait name (e.g., 'Vanguard') or original name if no mapping found
        """
        if not raw_name:
            return raw_name
        
        # Try exact match first
        if raw_name in self.traits_mapping:
            return self.traits_mapping[raw_name]
        
        # If no exact match, return original name
        return raw_name
    
    def map_item_name(self, raw_name: str) -> str:
        """
        Map a raw item name to a clean name.
        
        Args:
            raw_name: Raw item identifier from API
            
        Returns:
            Clean item name or original name if no mapping found
        """
        if not raw_name:
            return raw_name
        
        # Try exact match first
        if raw_name in self.items_mapping:
            return self.items_mapping[raw_name]
        
        # If no exact match, return original name
        return raw_name
    
    def map_unit_data(self, unit_data: Dict) -> Dict:
        """
        Apply mappings to a complete unit data structure.
        
        Args:
            unit_data: Unit dictionary from API response
            
        Returns:
            Unit dictionary with mapped names
        """
        if not isinstance(unit_data, dict):
            return unit_data
        
        # Create a copy to avoid modifying original
        mapped_unit = unit_data.copy()
        
        # Map character_id
        if 'character_id' in mapped_unit:
            mapped_unit['character_id'] = self.map_unit_name(mapped_unit['character_id'])
        
        # Map item names
        if 'itemNames' in mapped_unit and isinstance(mapped_unit['itemNames'], list):
            mapped_unit['item_names'] = [
                self.map_item_name(item) for item in mapped_unit['itemNames']
            ]
        elif 'item_names' in mapped_unit and isinstance(mapped_unit['item_names'], list):
            mapped_unit['item_names'] = [
                self.map_item_name(item) for item in mapped_unit['item_names']
            ]
        
        # Map unit name if present
        if 'name' in mapped_unit:
            mapped_unit['name'] = self.map_unit_name(mapped_unit['name'])
        
        return mapped_unit
    
    def map_trait_data(self, trait_data: Dict) -> Dict:
        """
        Apply mappings to a complete trait data structure.
        
        Args:
            trait_data: Trait dictionary from API response
            
        Returns:
            Trait dictionary with mapped names
        """
        if not isinstance(trait_data, dict):
            return trait_data
        
        # Create a copy to avoid modifying original
        mapped_trait = trait_data.copy()
        
        # Map trait name
        if 'name' in mapped_trait:
            mapped_trait['name'] = self.map_trait_name(mapped_trait['name'])
        
        return mapped_trait
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded mappings.
        
        Returns:
            Dictionary with mapping counts
        """
        return {
            'units': len(self.units_mapping),
            'traits': len(self.traits_mapping),
            'items': len(self.items_mapping)
        }

# Global mapper instance (will be initialized on first import)
_global_mapper: Optional[TFTNameMapper] = None

def get_mapper() -> TFTNameMapper:
    """
    Get the global name mapper instance.
    
    Returns:
        Singleton TFTNameMapper instance
    """
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = TFTNameMapper()
    return _global_mapper

def map_match_data(match_data: Dict) -> Dict:
    """
    Apply name mappings to a complete match data structure.
    
    Args:
        match_data: Complete match data from API
        
    Returns:
        Match data with all names mapped
    """
    mapper = get_mapper()
    
    if not isinstance(match_data, dict) or 'info' not in match_data:
        return match_data
    
    # Create a copy to avoid modifying original
    mapped_match = match_data.copy()
    mapped_match['info'] = match_data['info'].copy()
    
    # Map participant data
    if 'participants' in mapped_match['info']:
        mapped_participants = []
        for participant in mapped_match['info']['participants']:
            if not isinstance(participant, dict):
                mapped_participants.append(participant)
                continue
            
            mapped_participant = participant.copy()
            
            # Map units
            if 'units' in mapped_participant and isinstance(mapped_participant['units'], list):
                mapped_participant['units'] = [
                    mapper.map_unit_data(unit) for unit in mapped_participant['units']
                ]
            
            # Map traits
            if 'traits' in mapped_participant and isinstance(mapped_participant['traits'], list):
                mapped_participant['traits'] = [
                    mapper.map_trait_data(trait) for trait in mapped_participant['traits']
                ]
            
            mapped_participants.append(mapped_participant)
        
        mapped_match['info']['participants'] = mapped_participants
    
    return mapped_match

if __name__ == "__main__":
    # Test the mapper
    mapper = TFTNameMapper()
    stats = mapper.get_mapping_stats()
    print(f"Name Mapper Statistics: {stats}")
    
    # Test some mappings
    print(f"Raw 'TFT14_Jinx_34' -> '{mapper.map_unit_name('TFT14_Jinx_34')}'")
    print(f"Raw 'TFT14_Vanguard' -> '{mapper.map_trait_name('TFT14_Vanguard')}'")