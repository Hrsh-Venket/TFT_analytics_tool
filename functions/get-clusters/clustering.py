#!/usr/bin/env python3
"""
TFT BigQuery-Based Clustering System

Production-ready BigQuery-based hierarchical clustering for TFT compositions.
Designed for Firebase webapp integration with comprehensive cluster analysis.

Two-Level Clustering:
1. Sub-clusters: Exact carry matching for precise compositions
2. Main clusters: Groups of sub-clusters with 2-3 common carries

Carry Detection: Units with 2 or more items are considered carries.
"""

import logging
import json
import os
import numpy as np
from collections import Counter, defaultdict

from dataclasses import dataclass, asdict
from typing import List, Dict, Set, FrozenSet, Tuple, Optional, Any, Union
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test for scikit-learn availability
HAS_SKLEARN = False
try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    logger.warning("scikit-learn not available - clustering will use simplified mode")
    AgglomerativeClustering = None

# Test for BigQuery availability
HAS_BIGQUERY = False
try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    HAS_BIGQUERY = True
    logger.info("BigQuery dependencies available")
except ImportError:
    logger.warning("BigQuery dependencies not available - install google-cloud-bigquery")
    bigquery = None

# Test Mode Configuration
TEST_MODE = os.getenv('TFT_TEST_MODE', 'false').lower() == 'true'

# Clustering Configuration
CARRY_FREQUENCY_THRESHOLD = 0.75  # Minimum frequency for carries to be shown in main cluster names
CARRY_THRESHOLD = 0.75             # Minimum frequency for a unit to be considered a carry (75%)
GOLD_3STAR_THRESHOLD = 0.9        # Minimum frequency for g3star_ prefix (90%)
SILVER_3STAR_THRESHOLD = 0.5      # Minimum frequency for s3star_ prefix (50%)
TOP_UNITS_COUNT = 8               # Number of top units to display


@dataclass
class Composition:
    """Represents a TFT composition with metadata and carry information."""
    match_id: str
    puuid: str
    riot_id: str
    carries: FrozenSet[str]
    last_round: int
    placement: int
    level: int
    participant_data: dict
    sub_cluster_id: Optional[int] = None
    main_cluster_id: Optional[int] = None


@dataclass
class SubCluster:
    """Represents a sub-cluster of compositions with identical carry sets."""
    id: int
    carry_set: FrozenSet[str]
    compositions: List[Composition]
    size: int
    avg_placement: float
    winrate: float
    top4_rate: float


@dataclass
class MainCluster:
    """Represents a main cluster grouping sub-clusters with common carries."""
    id: int
    sub_cluster_ids: List[int]
    compositions: List[Composition]
    size: int
    avg_placement: float
    winrate: float
    top4_rate: float
    common_carries: List[str]
    top_units_display: str


class TFTClusteringEngine:
    """
    BigQuery-based two-level clustering engine for TFT compositions.
    
    Level 1: Sub-clusters based on exact carry matching
    Level 2: Main clusters grouping sub-clusters with 2-3 common carries
    
    Designed for Firebase webapp integration with comprehensive analysis capabilities.
    """
    
    def __init__(self, 
                 project_id: Optional[str] = None, 
                 dataset_id: str = 'tft_analytics',
                 min_sub_cluster_size: int = 5, 
                 min_main_cluster_size: int = 3):
        """
        Initialize the BigQuery TFT clustering engine.
        
        Args:
            project_id: GCP project ID (auto-detected if None)
            dataset_id: BigQuery dataset ID
            min_sub_cluster_size: Minimum size for valid sub-clusters
            min_main_cluster_size: Minimum size for valid main clusters
        """
        if not HAS_BIGQUERY and not TEST_MODE:
            raise ImportError("BigQuery dependencies not available. Install google-cloud-bigquery or enable TEST_MODE.")
        
        if HAS_BIGQUERY:
            self.client = bigquery.Client(project=project_id)
            self.project_id = project_id or self.client.project
        else:
            self.client = None
            self.project_id = project_id or "test-project"
        
        self.dataset_id = dataset_id
        self.table_id = f"{self.project_id}.{self.dataset_id}.match_participants"
        
        # Clustering parameters
        self.min_sub_cluster_size = min_sub_cluster_size
        self.min_main_cluster_size = min_main_cluster_size
        
        # Clustering state
        self.compositions: List[Composition] = []
        self.sub_clusters: List[SubCluster] = []
        self.main_clusters: List[MainCluster] = []
        self.main_cluster_assignments: Dict[int, int] = {}
        
        if TEST_MODE:
            logger.info("🧪 TFT Clustering running in TEST MODE")
    
    def extract_carry_units(self, participant: dict) -> FrozenSet[str]:
        """
        Extract carry units from a participant based on item count.
        
        Units with 2 or more items are considered carries.
        
        Args:
            participant: Participant data from match
            
        Returns:
            Set of carry unit IDs
        """
        carry_units = frozenset(
            unit['character_id'] for unit in participant.get('units', [])
            if len(unit.get('item_names', [])) >= 2
        )
        
        return carry_units
    
    def load_compositions_from_bigquery(self, 
                                        filters: Optional[Dict[str, Any]] = None,
                                        limit: Optional[int] = None) -> None:
        """Load and process compositions from BigQuery database."""
        logger.info("Loading compositions from BigQuery...")
        
        if TEST_MODE:
            self._load_test_compositions(limit or 1000)
            return
        
        try:
            # Build base query
            query = f"""
                SELECT 
                    match_id,
                    puuid,
                    riot_id_game_name,
                    riot_id_tagline,
                    placement,
                    level,
                    last_round,
                    units,
                    traits,
                    game_datetime,
                    tft_set_number
                FROM `{self.table_id}`
                WHERE 1=1
            """
            
            query_params = {}
            
            # Apply filters
            if filters:
                if 'set_number' in filters:
                    query += " AND tft_set_number = @set_number"
                    query_params['set_number'] = filters['set_number']
                
                if 'date_from' in filters:
                    query += " AND DATE(game_datetime) >= @date_from"
                    query_params['date_from'] = filters['date_from']
                
                if 'date_to' in filters:
                    query += " AND DATE(game_datetime) <= @date_to" 
                    query_params['date_to'] = filters['date_to']
                
                if 'placement_range' in filters:
                    min_place, max_place = filters['placement_range']
                    query += " AND placement >= @min_placement AND placement <= @max_placement"
                    query_params['min_placement'] = min_place
                    query_params['max_placement'] = max_place
            
            # Add limit
            if limit:
                query += f" LIMIT {limit}"
            
            # Configure query job with parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        key, 
                        "DATE" if key.startswith('date_') else "STRING" if isinstance(value, str) else "INTEGER", 
                        value
                    )
                    for key, value in query_params.items()
                ]
            )
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Process results into compositions
            compositions = []
            for row in results:
                # Extract participant data
                participant_data = {
                    'placement': row['placement'],
                    'level': row['level'],
                    'last_round': row['last_round'],
                    'units': row['units'] if row['units'] else [],
                    'traits': row['traits'] if row['traits'] else []
                }
                
                # Extract carries
                carries = self.extract_carry_units(participant_data)
                
                # Create composition
                riot_id = f"{row['riot_id_game_name'] or ''}#{row['riot_id_tagline'] or ''}"
                comp = Composition(
                    match_id=row['match_id'],
                    puuid=row['puuid'],
                    riot_id=riot_id,
                    carries=carries,
                    last_round=row['last_round'] or 50,
                    placement=row['placement'],
                    level=row['level'],
                    participant_data=participant_data
                )
                compositions.append(comp)
            
            self.compositions = compositions
            logger.info(f"Loaded {len(self.compositions)} compositions from BigQuery")
            
        except Exception as e:
            logger.error(f"Error loading compositions from BigQuery: {e}")
            if TEST_MODE:
                self._load_test_compositions(limit or 1000)
            else:
                raise
    
    def _load_test_compositions(self, count: int) -> None:
        """Load test compositions for development/testing."""
        logger.info(f"🧪 Loading {count} test compositions")
        
        # Generate realistic test data
        test_units = ['Jinx', 'Aphelios', 'Ezreal', 'Caitlyn', 'Twisted Fate', 'Zoe', 'Soraka', 'Syndra']
        test_items = ['Infinity Edge', 'Last Whisper', 'Guinsoo\'s Rageblade', 'Rapid Firecannon', 'Runaan\'s Hurricane']
        
        compositions = []
        for i in range(count):
            # Create varied carry patterns
            if i % 4 == 0:
                carries = frozenset(['Jinx', 'Aphelios'])  # Sniper carry
            elif i % 4 == 1:
                carries = frozenset(['Ezreal', 'Twisted Fate'])  # Arcane carry
            elif i % 4 == 2:
                carries = frozenset(['Jinx', 'Caitlyn'])  # Sniper variant
            else:
                carries = frozenset(['Syndra', 'Zoe'])  # Magic carry
            
            # Generate units with items
            units = []
            for unit_name in test_units[:6]:  # Take first 6 units
                item_count = 2 if unit_name in carries else np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
                unit_items = np.random.choice(test_items, size=min(item_count, 3), replace=False).tolist()
                
                units.append({
                    'character_id': unit_name,
                    'tier': np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1]),
                    'item_names': unit_items
                })
            
            # Generate participant data
            participant_data = {
                'placement': np.random.choice(range(1, 9), p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]),
                'level': np.random.choice(range(6, 10), p=[0.1, 0.3, 0.4, 0.2]),
                'last_round': np.random.choice(range(20, 45)),
                'units': units,
                'traits': [
                    {'name': 'Sniper', 'tier_current': 2 if 'Jinx' in carries else 1},
                    {'name': 'Star Guardian', 'tier_current': 2},
                ]
            }
            
            comp = Composition(
                match_id=f"TEST_MATCH_{i+1:04d}",
                puuid=f"test_player_{i+1}",
                riot_id=f"TestPlayer{i+1}#NA1",
                carries=carries,
                last_round=participant_data['last_round'],
                placement=participant_data['placement'],
                level=participant_data['level'],
                participant_data=participant_data
            )
            compositions.append(comp)
        
        self.compositions = compositions
        logger.info(f"Generated {len(self.compositions)} test compositions")
    
    def create_sub_clusters(self) -> None:
        """Create sub-clusters based on exact carry matching."""
        logger.info("Creating sub-clusters (exact carry matching)...")
        
        # Group compositions by identical carry sets
        carry_groups = defaultdict(list)
        for comp in self.compositions:
            carry_groups[comp.carries].append(comp)
        
        # Create sub-clusters from groups meeting minimum size
        sub_clusters = []
        sub_cluster_id = 0
        
        for carry_set, comps in carry_groups.items():
            if len(comps) >= self.min_sub_cluster_size:
                # Calculate statistics
                avg_placement = sum(c.placement for c in comps) / len(comps)
                winrate = sum(1 for c in comps if c.placement == 1) / len(comps) * 100
                top4_rate = sum(1 for c in comps if c.placement <= 4) / len(comps) * 100
                
                # Create sub-cluster
                sub_cluster = SubCluster(
                    id=sub_cluster_id,
                    carry_set=carry_set,
                    compositions=comps,
                    size=len(comps),
                    avg_placement=round(avg_placement, 2),
                    winrate=round(winrate, 2),
                    top4_rate=round(top4_rate, 2)
                )
                
                # Assign sub-cluster ID to compositions
                for comp in comps:
                    comp.sub_cluster_id = sub_cluster_id
                
                sub_clusters.append(sub_cluster)
                sub_cluster_id += 1
        
        self.sub_clusters = sub_clusters
        logger.info(f"Created {len(self.sub_clusters)} valid sub-clusters")
        logger.info(f"Sub-clustered {sum(sc.size for sc in self.sub_clusters)} compositions")
    
    def create_main_clusters(self) -> None:
        """Create main clusters by grouping sub-clusters with 2-3 common carries."""
        logger.info("Creating main clusters (2-3 common carries)...")
        
        if len(self.sub_clusters) < 2:
            logger.info("Not enough sub-clusters for main clustering")
            return
        
        # Build similarity matrix between sub-clusters
        n = len(self.sub_clusters)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_carry_similarity(
                    self.sub_clusters[i].carry_set,
                    self.sub_clusters[j].carry_set
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Convert similarity to distance matrix
        distance_matrix = 1.0 - similarity_matrix
        
        # Perform agglomerative clustering
        if HAS_SKLEARN:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                linkage='average',
                metric='precomputed',
                distance_threshold=0.4  # Requires at least 60% similarity
            )
            main_labels = clustering.fit_predict(distance_matrix)
        else:
            # Simplified clustering for test mode / when sklearn not available
            main_labels = self._simple_clustering(similarity_matrix)
        
        # Process main cluster assignments
        main_cluster_sizes = Counter(main_labels)
        valid_main_clusters = {
            label for label, size in main_cluster_sizes.items() 
            if size >= self.min_main_cluster_size
        }
        
        # Create main cluster objects
        main_clusters = []
        for main_cluster_id in valid_main_clusters:
            # Get sub-clusters in this main cluster
            sub_cluster_indices = [i for i, label in enumerate(main_labels) if label == main_cluster_id]
            sub_clusters_in_main = [self.sub_clusters[i] for i in sub_cluster_indices]
            
            # Get all compositions in this main cluster
            all_compositions = []
            for sub_cluster in sub_clusters_in_main:
                all_compositions.extend(sub_cluster.compositions)
            
            # Calculate main cluster statistics
            avg_placement = sum(c.placement for c in all_compositions) / len(all_compositions)
            winrate = sum(1 for c in all_compositions if c.placement == 1) / len(all_compositions) * 100
            top4_rate = sum(1 for c in all_compositions if c.placement <= 4) / len(all_compositions) * 100
            
            # Find common carries across sub-clusters
            common_carries = self._find_common_carries_in_main_cluster(sub_clusters_in_main)
            
            # Generate top units display
            top_units_display = self._analyze_unit_properties_in_cluster(all_compositions)
            
            # Create main cluster
            main_cluster = MainCluster(
                id=main_cluster_id,
                sub_cluster_ids=[sc.id for sc in sub_clusters_in_main],
                compositions=all_compositions,
                size=len(all_compositions),
                avg_placement=round(avg_placement, 2),
                winrate=round(winrate, 2),
                top4_rate=round(top4_rate, 2),
                common_carries=common_carries,
                top_units_display=top_units_display
            )
            
            main_clusters.append(main_cluster)
            
            # Update assignments
            for sub_cluster in sub_clusters_in_main:
                self.main_cluster_assignments[sub_cluster.id] = main_cluster_id
                
                # Propagate to compositions
                for comp in sub_cluster.compositions:
                    comp.main_cluster_id = main_cluster_id
        
        self.main_clusters = main_clusters
        logger.info(f"Created {len(valid_main_clusters)} main clusters")
        logger.info(f"Grouped {len([sc for sc in self.sub_clusters if sc.id in self.main_cluster_assignments])} sub-clusters")
    
    def _calculate_carry_similarity(self, carries1: FrozenSet[str], carries2: FrozenSet[str]) -> float:
        """
        Calculate similarity between two carry sets based on common units.
        
        Returns similarity score between 0.0 and 1.0 based on:
        - Common carries / max(len(carries1), len(carries2))
        - Bonus for having 2-3 common carries (ideal range)
        """
        if not carries1 or not carries2:
            return 0.0
        
        common_carries = carries1.intersection(carries2)
        common_count = len(common_carries)
        
        if common_count == 0:
            return 0.0
        
        # Base similarity: Jaccard coefficient
        union_size = len(carries1.union(carries2))
        jaccard = common_count / union_size if union_size > 0 else 0.0
        
        # Bonus for having 2-3 common carries (sweet spot for clustering)
        if 2 <= common_count <= 3:
            bonus = 0.3
        elif common_count == 1:
            bonus = 0.1
        else:
            bonus = -0.1  # Penalize too many or too few common carries
        
        return min(1.0, jaccard + bonus)
    
    def _simple_clustering(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Simple clustering algorithm for when sklearn is not available.
        Groups sub-clusters based on similarity threshold.
        """
        n = similarity_matrix.shape[0]
        labels = np.arange(n)  # Initially each sub-cluster is its own cluster
        threshold = 0.6  # 60% similarity threshold
        
        # Find pairs with high similarity and merge them
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    # Merge clusters: assign j's label to i's label
                    old_label = labels[j]
                    new_label = labels[i]
                    labels[labels == old_label] = new_label
        
        # Renumber labels to be consecutive
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        final_labels = np.array([label_map[label] for label in labels])
        
        return final_labels
    
    def _find_common_carries_in_main_cluster(self, sub_clusters: List[SubCluster], frequency_threshold: float = 0.7) -> List[str]:
        """Find carries that appear frequently across sub-clusters in a main cluster."""
        if not sub_clusters:
            return []
        
        # Count how many sub-clusters each carry appears in
        carry_counts = defaultdict(int)
        total_sub_clusters = len(sub_clusters)
        
        for sub_cluster in sub_clusters:
            for carry in sub_cluster.carry_set:
                carry_counts[carry] += 1
        
        # Filter carries that meet frequency threshold
        common_carries = []
        for carry, count in carry_counts.items():
            frequency = count / total_sub_clusters
            if frequency >= frequency_threshold:
                common_carries.append(carry)
        
        return sorted(common_carries)
    
    def _analyze_unit_properties_in_cluster(self, compositions: List[Composition]) -> str:
        """
        Analyze unit frequencies and properties within a cluster and return formatted display string.
        
        Args:
            compositions: List of compositions in the cluster
            
        Returns:
            Formatted string showing top units with prefixes
        """
        if not compositions:
            return ""
        
        total_matches = len(compositions)
        unit_stats = defaultdict(lambda: {
            'frequency': 0,
            'carry_count': 0,
            'star_counts': defaultdict(int)
        })
        
        # Analyze each composition
        for comp in compositions:
            units_in_comp = set()
            
            for unit in comp.participant_data.get('units', []):
                char_id = unit.get('character_id', '')
                if not char_id:
                    continue
                    
                units_in_comp.add(char_id)
                
                # Count star level
                tier = unit.get('tier', 1)
                unit_stats[char_id]['star_counts'][tier] += 1
                
                # Count if it's a carry (2+ items)
                item_count = len(unit.get('item_names', []))
                if item_count >= 2:
                    unit_stats[char_id]['carry_count'] += 1
            
            # Count frequency (presence in match, not duplicates)
            for unit_name in units_in_comp:
                unit_stats[unit_name]['frequency'] += 1
        
        # Calculate percentages and create display names
        unit_display_list = []
        
        for unit_name, stats in unit_stats.items():
            frequency = stats['frequency'] / total_matches
            carry_frequency = stats['carry_count'] / total_matches if stats['frequency'] > 0 else 0
            
            # Calculate 3-star frequency
            star3_count = stats['star_counts'][3]
            star3_frequency = star3_count / stats['frequency'] if stats['frequency'] > 0 else 0
            
            # Build display name with prefixes
            display_name = unit_name
            
            # Add star prefixes (gold takes priority over silver)
            if star3_frequency >= GOLD_3STAR_THRESHOLD:
                display_name = f"g3star_{display_name}"
            elif star3_frequency >= SILVER_3STAR_THRESHOLD:
                display_name = f"s3star_{display_name}"
            
            # Add carry prefix
            if carry_frequency >= CARRY_THRESHOLD:
                display_name = f"Carry_{display_name}"
            
            unit_display_list.append((display_name, frequency))
        
        # Sort by frequency and take top N
        unit_display_list.sort(key=lambda x: x[1], reverse=True)
        top_units = unit_display_list[:TOP_UNITS_COUNT]
        
        return ', '.join([unit[0] for unit in top_units])
    
    def get_clustering_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive clustering statistics."""
        total_compositions = len(self.compositions)
        sub_clustered = len([c for c in self.compositions if c.sub_cluster_id is not None])
        main_clustered = len([c for c in self.compositions if c.main_cluster_id is not None])
        
        # Sub-cluster statistics
        sub_cluster_sizes = [sc.size for sc in self.sub_clusters]
        
        # Main cluster statistics  
        main_cluster_sizes = [mc.size for mc in self.main_clusters]
        
        return {
            'total_compositions': total_compositions,
            'sub_clusters': {
                'count': len(self.sub_clusters),
                'compositions_clustered': sub_clustered,
                'avg_size': round(np.mean(sub_cluster_sizes), 1) if sub_cluster_sizes else 0,
                'largest_size': max(sub_cluster_sizes) if sub_cluster_sizes else 0,
                'coverage': round((sub_clustered / total_compositions) * 100, 1) if total_compositions > 0 else 0
            },
            'main_clusters': {
                'count': len(self.main_clusters),
                'sub_clusters_grouped': len(self.main_cluster_assignments),
                'compositions_clustered': main_clustered,
                'avg_size': round(np.mean(main_cluster_sizes), 1) if main_cluster_sizes else 0,
                'largest_size': max(main_cluster_sizes) if main_cluster_sizes else 0,
                'coverage': round((main_clustered / total_compositions) * 100, 1) if total_compositions > 0 else 0
            }
        }
    
    def get_cluster_summary(self, cluster_type: str = 'main', top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get summary of top clusters sorted by performance.
        
        Args:
            cluster_type: 'main' or 'sub'
            top_n: Number of top clusters to return
            
        Returns:
            List of cluster summary dictionaries
        """
        if cluster_type == 'main':
            clusters_data = []
            for cluster in sorted(self.main_clusters, key=lambda x: x.avg_placement):
                clusters_data.append({
                    'id': cluster.id,
                    'type': 'main',
                    'size': cluster.size,
                    'avg_placement': cluster.avg_placement,
                    'winrate': cluster.winrate,
                    'top4_rate': cluster.top4_rate,
                    'common_carries': cluster.common_carries,
                    'top_units': cluster.top_units_display,
                    'sub_cluster_count': len(cluster.sub_cluster_ids)
                })
        else:  # sub-clusters
            clusters_data = []
            for cluster in sorted(self.sub_clusters, key=lambda x: x.avg_placement):
                clusters_data.append({
                    'id': cluster.id,
                    'type': 'sub',
                    'size': cluster.size,
                    'avg_placement': cluster.avg_placement,
                    'winrate': cluster.winrate,
                    'top4_rate': cluster.top4_rate,
                    'carries': list(cluster.carry_set),
                    'main_cluster_id': self.main_cluster_assignments.get(cluster.id, -1)
                })
        
        return clusters_data[:top_n]
    
    def get_cluster_details(self, cluster_id: int, cluster_type: str = 'main') -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            cluster_type: 'main' or 'sub'
            
        Returns:
            Detailed cluster information or None if not found
        """
        if cluster_type == 'main':
            for cluster in self.main_clusters:
                if cluster.id == cluster_id:
                    return {
                        'id': cluster.id,
                        'type': 'main',
                        'size': cluster.size,
                        'avg_placement': cluster.avg_placement,
                        'winrate': cluster.winrate,
                        'top4_rate': cluster.top4_rate,
                        'common_carries': cluster.common_carries,
                        'top_units': cluster.top_units_display,
                        'sub_clusters': [
                            {
                                'id': sc.id,
                                'size': sc.size,
                                'avg_placement': sc.avg_placement,
                                'carries': list(sc.carry_set)
                            }
                            for sc in self.sub_clusters 
                            if sc.id in cluster.sub_cluster_ids
                        ],
                        'sample_compositions': [
                            {
                                'match_id': comp.match_id,
                                'riot_id': comp.riot_id,
                                'placement': comp.placement,
                                'level': comp.level,
                                'carries': list(comp.carries)
                            }
                            for comp in cluster.compositions[:5]  # First 5 samples
                        ]
                    }
        else:  # sub-cluster
            for cluster in self.sub_clusters:
                if cluster.id == cluster_id:
                    return {
                        'id': cluster.id,
                        'type': 'sub',
                        'size': cluster.size,
                        'avg_placement': cluster.avg_placement,
                        'winrate': cluster.winrate,
                        'top4_rate': cluster.top4_rate,
                        'carries': list(cluster.carry_set),
                        'main_cluster_id': self.main_cluster_assignments.get(cluster.id, -1),
                        'sample_compositions': [
                            {
                                'match_id': comp.match_id,
                                'riot_id': comp.riot_id,
                                'placement': comp.placement,
                                'level': comp.level,
                                'carries': list(comp.carries)
                            }
                            for comp in cluster.compositions[:5]  # First 5 samples
                        ]
                    }
        
        return None


def run_clustering_analysis(project_id: Optional[str] = None,
                          dataset_id: str = 'tft_analytics',
                          filters: Optional[Dict[str, Any]] = None,
                          min_sub_cluster_size: int = 5,
                          min_main_cluster_size: int = 3,
                          limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Run complete clustering analysis on BigQuery data.
    
    Args:
        project_id: GCP project ID (auto-detected if None)
        dataset_id: BigQuery dataset ID
        filters: Optional filters for data selection
        min_sub_cluster_size: Minimum size for valid sub-clusters
        min_main_cluster_size: Minimum size for valid main clusters
        limit: Optional limit on compositions to analyze
        
    Returns:
        Dictionary with clustering results and statistics
    """
    logger.info("Starting TFT BigQuery Clustering Analysis")
    logger.info("=" * 50)
    
    # Initialize clustering engine
    engine = TFTClusteringEngine(
        project_id=project_id,
        dataset_id=dataset_id,
        min_sub_cluster_size=min_sub_cluster_size,
        min_main_cluster_size=min_main_cluster_size
    )
    
    try:
        # Execute clustering pipeline
        engine.load_compositions_from_bigquery(filters=filters, limit=limit)
        engine.create_sub_clusters()
        engine.create_main_clusters()
        
        # Generate statistics
        stats = engine.get_clustering_statistics()
        
        # Get cluster summaries
        main_cluster_summary = engine.get_cluster_summary('main', top_n=10)
        sub_cluster_summary = engine.get_cluster_summary('sub', top_n=20)
        
        # Compile results
        results = {
            'statistics': stats,
            'main_clusters': main_cluster_summary,
            'sub_clusters': sub_cluster_summary,
            'engine': engine  # Include engine for further analysis
        }
        
        # Print summary
        logger.info("\nClustering Results Summary:")
        logger.info(f"Total compositions: {stats['total_compositions']}")
        logger.info(f"Sub-clusters: {stats['sub_clusters']['count']} (avg size: {stats['sub_clusters']['avg_size']})")
        logger.info(f"Main clusters: {stats['main_clusters']['count']} (avg size: {stats['main_clusters']['avg_size']})")
        logger.info(f"Sub-cluster coverage: {stats['sub_clusters']['coverage']}%")
        logger.info(f"Main cluster coverage: {stats['main_clusters']['coverage']}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        raise


def test_connection() -> Dict[str, Any]:
    """
    Test BigQuery connection and clustering system status.
    
    Returns:
        Dictionary with connection status and basic information
    """
    try:
        if not HAS_BIGQUERY:
            if TEST_MODE:
                # In test mode, simulate successful connection
                return {
                    'success': True,
                    'message': 'Test mode - BigQuery clustering simulation active',
                    'project_id': 'test-project',
                    'dataset_id': 'tft_analytics',
                    'table_id': 'test-project.tft_analytics.match_participants'
                }
            else:
                return {
                    'success': False,
                    'error': 'BigQuery dependencies not available',
                    'message': 'Install google-cloud-bigquery'
                }
        
        # Test with small sample
        engine = TFTClusteringEngine()
        engine.load_compositions_from_bigquery(limit=10)
        
        return {
            'success': True,
            'message': 'BigQuery clustering connection successful',
            'compositions_loaded': len(engine.compositions),
            'project_id': engine.project_id,
            'dataset_id': engine.dataset_id,
            'table_id': engine.table_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'BigQuery clustering connection failed'
        }


def main():
    """Main function for testing the clustering system."""
    print("TFT Analytics BigQuery Clustering System")
    print("=" * 50)
    
    # Test connection
    print("Testing BigQuery clustering connection...")
    connection_result = test_connection()
    
    if connection_result['success']:
        print("[SUCCESS] BigQuery clustering connection successful")
        print(f"   Message: {connection_result['message']}")
        print(f"   Project: {connection_result['project_id']}")
        print(f"   Dataset: {connection_result['dataset_id']}")
    else:
        print("[ERROR] BigQuery clustering connection failed")
        print(f"   Error: {connection_result['error']}")
        if not TEST_MODE:
            print("   Enable test mode with: set TFT_TEST_MODE=true")
            return
    
    print("\nTesting clustering functionality...")
    
    # Test clustering with small sample
    try:
        print("\n1. Running clustering analysis...")
        results = run_clustering_analysis(limit=200 if not TEST_MODE else None)
        
        stats = results['statistics']
        print("[SUCCESS] Clustering analysis completed:")
        print(f"   Total compositions: {stats['total_compositions']}")
        print(f"   Sub-clusters created: {stats['sub_clusters']['count']}")
        print(f"   Main clusters created: {stats['main_clusters']['count']}")
        print(f"   Coverage: {stats['sub_clusters']['coverage']}% sub-clustered, {stats['main_clusters']['coverage']}% main-clustered")
        
        # Show top main clusters
        if results['main_clusters']:
            print(f"\n2. Top 3 Main Clusters (by performance):")
            for i, cluster in enumerate(results['main_clusters'][:3], 1):
                print(f"   {i}. Cluster {cluster['id']}: {cluster['size']} comps, {cluster['avg_placement']:.2f} avg place")
                print(f"      Common carries: {cluster['common_carries']}")
                print(f"      Top units: {cluster['top_units'][:60]}...")
        
        # Test detailed cluster information
        if results['main_clusters']:
            print(f"\n3. Detailed cluster information test...")
            cluster_id = results['main_clusters'][0]['id']
            engine = results['engine']
            
            details = engine.get_cluster_details(cluster_id, 'main')
            if details:
                print(f"[SUCCESS] Retrieved details for main cluster {cluster_id}:")
                print(f"   Sub-clusters: {len(details['sub_clusters'])}")
                print(f"   Sample compositions: {len(details['sample_compositions'])}")
            
    except Exception as e:
        print(f"[ERROR] Clustering analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Clustering system testing completed!")
    
    print("\nFirebase Integration Examples:")
    print("=" * 50)
    print("```python")
    print("# Firebase Cloud Function example")
    print("from clustering import run_clustering_analysis, TFTClusteringEngine")
    print("")
    print("def get_cluster_analysis(request):")
    print("    filters = request.json.get('filters', {})")
    print("    results = run_clustering_analysis(filters=filters, limit=1000)")
    print("    return {'clusters': results['main_clusters'], 'stats': results['statistics']}")
    print("")
    print("def get_cluster_details(request):")
    print("    cluster_id = request.json.get('cluster_id')")
    print("    cluster_type = request.json.get('type', 'main')")
    print("    ")
    print("    engine = TFTClusteringEngine()")
    print("    engine.load_compositions_from_bigquery()")
    print("    engine.create_sub_clusters()")
    print("    engine.create_main_clusters()")
    print("    ")
    print("    details = engine.get_cluster_details(cluster_id, cluster_type)")
    print("    return {'cluster': details}")
    print("```")


if __name__ == "__main__":
    main()