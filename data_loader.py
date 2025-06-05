import pandas as pd
import os
from pathlib import Path
import threading
import time
import numpy as np
class DataLoader:
    def __init__(self, data_dir):
        """
        Initialize the data loader with the directory containing CSV files
        
        Args:
            data_dir (Path): Directory path containing the CSV files
        """
        self.data_dir = Path(data_dir)
        self.file_paths = {
            'experiment_results': self.data_dir / 'experiment_results.csv',
            'global_gradients': self.data_dir / 'global_gradients.csv',
            'group_scores': self.data_dir / 'group_scores.csv',
            'summary_stats': self.data_dir / 'summary_stats.csv',
            'user_membership': self.data_dir / 'user_membership.csv',
            'user_scores': self.data_dir / 'user_scores.csv'
        }
        
        # Data cache
        self._cache = {
            'experiment_results': None,
            'global_gradients': None,
            'group_scores': None,
            'summary_stats': None,
            'user_membership': None,
            'user_scores': None
        }
        
        # Current selected experiment ID
        self.current_experiment_id = None
        # Separate experiment ID for HierarchicalFL
        self.hierarchical_experiment_id = None
        # Flag to use separate experiment ID for HierarchicalFL
        self.use_separate_hierarchical = False
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load data initially
        self.load_all_data()
    
    def load_all_data(self):
        """Load all data files with thread safety"""
        with self._lock:
            for file_key, file_path in self.file_paths.items():
                try:
                    if file_path.exists():
                        self._cache[file_key] = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error loading {file_key}: {e}")
            
            # Set the default experiment ID if not already set
            if self.current_experiment_id is None and self._cache['experiment_results'] is not None:
                if 'experiment_id' in self._cache['experiment_results'].columns:
                    self.current_experiment_id = self._cache['experiment_results']['experiment_id'].max()
                    self.hierarchical_experiment_id = self.current_experiment_id
                    print(f"Setting default experiment ID to {self.current_experiment_id}")
    
    def get_experiment_results(self, experiment_id=None):
        """
        Get experiment results data, optionally filtered by experiment ID
        
        Args:
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Filtered experiment results
        """
        with self._lock:
            if self._cache['experiment_results'] is None:
                self.load_file('experiment_results')
            
            if self._cache['experiment_results'] is None:
                return None
            
            # Special case: if we're using separate hierarchical experiment ID
            # and no specific experiment_id was provided
            if experiment_id is None and self.use_separate_hierarchical:
                return self.get_mixed_data('experiment_results')
                
            # Filter by experiment ID if provided
            if experiment_id is not None and 'experiment_id' in self._cache['experiment_results'].columns:
                return self._cache['experiment_results'][self._cache['experiment_results']['experiment_id'] == experiment_id]
            
            # Use current experiment ID if set
            if self.current_experiment_id is not None and 'experiment_id' in self._cache['experiment_results'].columns:
                return self._cache['experiment_results'][self._cache['experiment_results']['experiment_id'] == self.current_experiment_id]
            
            return self._cache['experiment_results']
            
    def get_available_experiment_ids(self):
        """
        Get list of available experiment IDs
        
        Returns:
            list: Available experiment IDs
        """
        with self._lock:
            if self._cache['experiment_results'] is None:
                self.load_file('experiment_results')
            
            if self._cache['experiment_results'] is None or 'experiment_id' not in self._cache['experiment_results'].columns:
                return []
            
            return sorted(self._cache['experiment_results']['experiment_id'].unique())
    
    def get_global_gradients(self, experiment_id=None):
        """
        Get global gradients data, optionally filtered by experiment ID
        
        Args:
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Filtered global gradients
        """
        with self._lock:
            if self._cache['global_gradients'] is None:
                self.load_file('global_gradients')
            
            data = self._cache['global_gradients']
            if data is None:
                return None
            
            # Special case: if we're using separate hierarchical experiment ID
            # and no specific experiment_id was provided
            if experiment_id is None and self.use_separate_hierarchical and 'aggregation_name' in data.columns:
                return self.get_mixed_data('global_gradients')
                
            # Filter by experiment ID if provided
            if experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == experiment_id]
            
            # Use current experiment ID if set
            if self.current_experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == self.current_experiment_id]
            
            return data
    
    def get_group_scores(self, experiment_id=None):
        """
        Get group scores data, optionally filtered by experiment ID
        
        Args:
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Filtered group scores
        """
        with self._lock:
            if self._cache['group_scores'] is None:
                self.load_file('group_scores')
            
            data = self._cache['group_scores']
            if data is None:
                return None
            
            # Special case: if we're using separate hierarchical experiment ID
            # and no specific experiment_id was provided
            if experiment_id is None and self.use_separate_hierarchical and 'aggregation_name' in data.columns:
                return self.get_mixed_data('group_scores')
                
            # Filter by experiment ID if provided
            if experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == experiment_id]
            
            # Use current experiment ID if set
            if self.current_experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == self.current_experiment_id]
            
            return data
    
    def get_summary_stats(self, experiment_id=None):
        """
        Get summary statistics data, optionally filtered by experiment ID
        
        Args:
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Filtered summary statistics
        """
        with self._lock:
            if self._cache['summary_stats'] is None:
                self.load_file('summary_stats')
            
            data = self._cache['summary_stats']
            if data is None:
                return None
            
            # Special case: if we're using separate hierarchical experiment ID
            # and no specific experiment_id was provided
            if experiment_id is None and self.use_separate_hierarchical and 'aggregation_name' in data.columns:
                return self.get_mixed_data('summary_stats')
                
            # Filter by experiment ID if provided
            if experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == experiment_id]
            
            # Use current experiment ID if set
            if self.current_experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == self.current_experiment_id]
            
            return data
    
    def get_user_membership(self, experiment_id=None):
        """
        Get user membership data, optionally filtered by experiment ID
        
        Args:
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Filtered user membership
        """
        with self._lock:
            if self._cache['user_membership'] is None:
                self.load_file('user_membership')
            
            data = self._cache['user_membership']
            if data is None:
                return None
            
            # Special case: if we're using separate hierarchical experiment ID
            # and no specific experiment_id was provided
            if experiment_id is None and self.use_separate_hierarchical and 'aggregation_name' in data.columns:
                return self.get_mixed_data('user_membership')
                
            # Filter by experiment ID if provided
            if experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == experiment_id]
            
            # Use current experiment ID if set
            if self.current_experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == self.current_experiment_id]
            
            return data
    
    def get_user_scores(self, experiment_id=None):
        """
        Get user scores data, optionally filtered by experiment ID
        
        Args:
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Filtered user scores
        """
        with self._lock:
            if self._cache['user_scores'] is None:
                self.load_file('user_scores')
            
            data = self._cache['user_scores']
            if data is None:
                return None
            
            # Special case: if we're using separate hierarchical experiment ID
            # and no specific experiment_id was provided
            if experiment_id is None and self.use_separate_hierarchical and 'aggregation_name' in data.columns:
                return self.get_mixed_data('user_scores')
                
            # Filter by experiment ID if provided
            if experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == experiment_id]
            
            # Use current experiment ID if set
            if self.current_experiment_id is not None and 'experiment_id' in data.columns:
                return data[data['experiment_id'] == self.current_experiment_id]
            
            return data
    
    def load_file(self, file_key):
        """
        Load a specific file with thread safety
        
        Args:
            file_key (str): Key of the file to load
        """
        with self._lock:
            try:
                if self.file_paths[file_key].exists():
                    self._cache[file_key] = pd.read_csv(self.file_paths[file_key])
            except Exception as e:
                print(f"Error loading {file_key}: {e}")
    
    def get_latest_round(self):
        """Get the latest round number from available data"""
        latest_round = 0
        
        with self._lock:
            # Check summary stats first
            if self._cache['summary_stats'] is not None and not self._cache['summary_stats'].empty:
                latest_round = max(latest_round, self._cache['summary_stats']['round'].max())
            
            # Check global gradients
            if self._cache['global_gradients'] is not None and not self._cache['global_gradients'].empty:
                latest_round = max(latest_round, self._cache['global_gradients']['round'].max())
            
            # Check experiment results
            if self._cache['experiment_results'] is not None and not self._cache['experiment_results'].empty:
                latest_round = max(latest_round, self._cache['experiment_results']['round_num'].max())
                
        return latest_round
    
    def get_data_by_round(self, data_key, round_num=None, experiment_id=None):
        """
        Get data for a specific round or all rounds, optionally filtered by experiment ID
        
        Args:
            data_key (str): Key of the data to retrieve
            round_num (int, optional): Round number to filter by
            experiment_id (int, optional): Experiment ID to filter by
        
        Returns:
            pandas.DataFrame: Data filtered by round and/or experiment ID if specified
        """
        with self._lock:
            data = self._cache.get(data_key)
            if data is None or data.empty:
                return None
            
            # Create a copy to avoid modifying the original
            filtered_data = data.copy()
            
            # Handle experiment ID filtering if applicable
            if experiment_id is not None and 'experiment_id' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['experiment_id'] == experiment_id]
            elif self.current_experiment_id is not None and 'experiment_id' in filtered_data.columns:
                # Use mixed experiment handling when appropriate
                if self.use_separate_hierarchical and 'aggregation_name' in filtered_data.columns:
                    # Get hierarchical data
                    hierarchical_data = filtered_data[
                        (filtered_data['experiment_id'] == self.hierarchical_experiment_id) &
                        (filtered_data['aggregation_name'].str.lower().str.contains('hierarchical') | 
                         filtered_data['aggregation_name'].str.lower().str.contains('heirichal'))
                    ]
                    
                    # Get other data
                    other_data = filtered_data[
                        (filtered_data['experiment_id'] == self.current_experiment_id) &
                        ~(filtered_data['aggregation_name'].str.lower().str.contains('hierarchical') | 
                          filtered_data['aggregation_name'].str.lower().str.contains('heirichal'))
                    ]
                    
                    # Combine
                    filtered_data = pd.concat([hierarchical_data, other_data])
                else:
                    filtered_data = filtered_data[filtered_data['experiment_id'] == self.current_experiment_id]
            
            # Handle round filtering
            if round_num is not None:
                # Handle different column names for round
                round_col = 'round_num' if data_key == 'experiment_results' else 'round'
                if round_col in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[round_col] == round_num]
            
            return filtered_data
    
    def set_current_experiment_id(self, experiment_id):
        """
        Set the current experiment ID for filtering data
        
        Args:
            experiment_id (int): Experiment ID to set as current
        """
        with self._lock:
            self.current_experiment_id = experiment_id
            # When not using separate hierarchical experiment ID, set it to match the current one
            if not self.use_separate_hierarchical:
                self.hierarchical_experiment_id = experiment_id
            print(f"Current experiment ID set to {experiment_id}")

    def set_hierarchical_experiment_id(self, experiment_id):
        """
        Set the experiment ID specifically for HierarchicalFL
        
        Args:
            experiment_id (int): Experiment ID to use for HierarchicalFL data
        """
        with self._lock:
            self.hierarchical_experiment_id = experiment_id
            self.use_separate_hierarchical = True
            print(f"HierarchicalFL experiment ID set to {experiment_id}")

    def use_same_experiment_id(self):
        """
        Configure the system to use the same experiment ID for all data
        """
        with self._lock:
            self.use_separate_hierarchical = False
            self.hierarchical_experiment_id = self.current_experiment_id
            print(f"Now using the same experiment ID ({self.current_experiment_id}) for all data")

    def get_mixed_data(self, data_key):
        """
        Generic function to get mixed data from two different experiment IDs
        
        Args:
            data_key (str): The key for the data cache (e.g., 'experiment_results', 'global_gradients')
            
        Returns:
            pandas.DataFrame: Combined data from both experiment IDs
        """
        if self._cache[data_key] is None:
            self.load_file(data_key)
            
        if self._cache[data_key] is None:
            return None
            
        data = self._cache[data_key]
        
        if 'experiment_id' not in data.columns or 'aggregation_name' not in data.columns:
            return data
            
        hierarchical_data = None
        other_data = None
        
        # Get HierarchicalFL data from hierarchical_experiment_id
        hierarchical_data = data[
            (data['experiment_id'] == self.hierarchical_experiment_id) &
            (data['aggregation_name'].str.lower().str.contains('hierarchical') | 
             data['aggregation_name'].str.lower().str.contains('heirichal'))
        ]
        
        # Get other aggregation rule data from current_experiment_id
        other_data = data[
            (data['experiment_id'] == self.current_experiment_id) &
            ~(data['aggregation_name'].str.lower().str.contains('hierarchical') | 
              data['aggregation_name'].str.lower().str.contains('heirichal'))
        ]
        
        # Combine the dataframes
        combined_data = pd.concat([hierarchical_data, other_data])
        
        return combined_data

    def get_protocol_comparison_data(self, experiment_id=None, use_max_accuracy=False):
        """
        Get formatted protocol comparison data with final round results for each configuration

        Args:
            experiment_id (int, optional): Experiment ID to filter by. If None, uses current_experiment_id
            use_max_accuracy (bool, optional): Whether to use the maximum accuracy across all rounds
                                              instead of the final round accuracy. Defaults to False.

        Returns:
            dict: Dictionary containing processed comparison data or None if no data available
        """
        with self._lock:
            # Get experiment results
            results = self.get_experiment_results(experiment_id)

            if results is None or results.empty:
                return None

            # Determine round column name
            round_col = 'round_num' if 'round_num' in results.columns else 'round'

            # Get unique protocols
            protocols = []
            if 'aggregation_name' in results.columns:
                protocols = sorted(results['aggregation_name'].unique())
            else:
                return None  # No protocols to compare

            # Get attack types
            attack_types = []
            has_no_attack = False

            if 'malicious_type' in results.columns:
                attack_types = sorted(results['malicious_type'].dropna().unique())
                has_no_attack = results['malicious_type'].isnull().any()

            if has_no_attack:
                attack_types = ['No Attack'] + attack_types
            elif not attack_types:
                attack_types = ['No Attack']

            # Get bias values
            bias_values = []
            if 'bias_values' in results.columns:
                bias_values = sorted(results['bias_values'].unique())
            else:
                bias_values = [0]

            # Get unique experiment configurations
            results['experiment_config'] = results.apply(
                lambda row: f"{row['aggregation_name']}_{str(row.get('malicious_type', 'None'))}_{row.get('bias_values', 0)}",
                axis=1
            )

            # Process data based on the selected approach
            if use_max_accuracy:
                # For max accuracy, find the round with maximum accuracy for each experiment config
                if 'round_accuracy' in results.columns:
                    # Group by experiment_config and find the round with max accuracy
                    metrics_of_interest = ['round_accuracy']
                    if 'round_backdoor_success' in results.columns:
                        metrics_of_interest.append('round_backdoor_success')

                    # Create a copy for working with
                    max_acc_rounds = results.copy()

                    # First, group by experiment_config and get the max accuracy
                    max_accs = max_acc_rounds.groupby('experiment_config')['round_accuracy'].max().reset_index()

                    # Now join back to find the corresponding rows
                    final_rounds = pd.merge(
                        results,
                        max_accs,
                        on=['experiment_config', 'round_accuracy'],
                        how='inner'
                    )

                    # If there are multiple rounds with the same max accuracy, take the latest
                    final_rounds = final_rounds.sort_values(by=[round_col], ascending=False)
                    final_rounds = final_rounds.drop_duplicates(subset=['experiment_config'])

                    # Add indicator that max accuracy was used
                    final_rounds['accuracy_type'] = 'maximum'
                else:
                    # Fallback to final round if round_accuracy is not available
                    max_rounds = results.groupby('experiment_config')[round_col].max().reset_index()
                    final_rounds = pd.merge(
                        results,
                        max_rounds,
                        on=['experiment_config', round_col],
                        how='inner'
                    )
                    # Add indicator that final round was used
                    final_rounds['accuracy_type'] = 'final_round'
            else:
                # For final accuracy, use the final round of each experiment
                max_rounds = results.groupby('experiment_config')[round_col].max().reset_index()
                final_rounds = pd.merge(
                    results,
                    max_rounds,
                    on=['experiment_config', round_col],
                    how='inner'
                )
                # Add indicator that final round was used
                final_rounds['accuracy_type'] = 'final_round'
            
            # Prepare comparison data structure
            comparison_data = {}
            
            # Process data for each attack type
            for attack_type in attack_types:
                comparison_data[attack_type] = {}
                
                # Filter data for this attack type
                if attack_type == 'No Attack':
                    attack_data = final_rounds[final_rounds['malicious_type'].isnull()]
                else:
                    attack_data = final_rounds[final_rounds['malicious_type'] == attack_type]
                    
                if attack_data.empty:
                    continue
                    
                # Process metrics
                metrics = ['round_accuracy']
                if 'round_backdoor_success' in attack_data.columns and attack_type != 'No Attack':
                    metrics.append('round_backdoor_success')
                    
                for metric in metrics:
                    if metric not in attack_data.columns:
                        continue
                        
                    comparison_data[attack_type][metric] = {}
                    
                    # Create data structure for each bias value
                    for bias in bias_values:
                        comparison_data[attack_type][metric][bias] = {}
                        bias_data = attack_data[attack_data['bias_values'] == bias]
                        
                        if bias_data.empty:
                            continue
                            
                        # Get data for each protocol
                        for protocol in protocols:
                            protocol_data = bias_data[bias_data['aggregation_name'] == protocol]
                            if not protocol_data.empty:
                                comparison_data[attack_type][metric][bias][protocol] = protocol_data[metric].mean()
            
            return {
                'results': results,
                'final_rounds': final_rounds,
                'protocols': protocols,
                'attack_types': attack_types,
                'bias_values': bias_values,
                'round_col': round_col,
                'comparison_data': comparison_data
            }
        
    def get_best_protocols(self, experiment_id=None, metric='round_accuracy', is_higher_better=True, use_max_accuracy=False):
        """
        Identify the best performing protocol for each attack type and bias value

        Args:
            experiment_id (int, optional): Experiment ID to filter by
            metric (str): Metric to evaluate (e.g., 'round_accuracy', 'round_backdoor_success')
            is_higher_better (bool): Whether higher values are better (True for accuracy, False for backdoor)
            use_max_accuracy (bool, optional): Whether to use the maximum accuracy across all rounds
                                               instead of the final round accuracy. Defaults to False.

        Returns:
            dict: Dictionary mapping (attack_type, bias_value) to best protocol and score
        """
        comparison_data = self.get_protocol_comparison_data(experiment_id, use_max_accuracy=use_max_accuracy)
        if not comparison_data:
            return {}
            
        best_protocols = {}
        
        for attack_type in comparison_data['attack_types']:
            if attack_type not in comparison_data['comparison_data']:
                continue
                
            attack_data = comparison_data['comparison_data'][attack_type]
            if metric not in attack_data:
                continue
                
            for bias in comparison_data['bias_values']:
                if bias not in attack_data[metric]:
                    continue
                    
                bias_data = attack_data[metric][bias]
                if not bias_data:
                    continue
                    
                # Find best protocol
                best_protocol = None
                best_score = None
                
                for protocol, score in bias_data.items():
                    if best_score is None or (is_higher_better and score > best_score) or (not is_higher_better and score < best_score):
                        best_score = score
                        best_protocol = protocol
                
                if best_protocol:
                    best_protocols[(attack_type, bias)] = {
                        'protocol': best_protocol,
                        'score': best_score
                    }
        
        return best_protocols
    

    def get_protocol_rankings(self, experiment_id=None, metric='round_accuracy', is_higher_better=True, use_max_accuracy=False):
        """
        Rank protocols by their average performance across attack types and bias values

        Args:
            experiment_id (int, optional): Experiment ID to filter by
            metric (str): Metric to evaluate (e.g., 'round_accuracy', 'round_backdoor_success')
            is_higher_better (bool): Whether higher values are better (True for accuracy, False for backdoor)
            use_max_accuracy (bool, optional): Whether to use the maximum accuracy across all rounds
                                               instead of the final round accuracy. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with protocol rankings and statistics
        """
        comparison_data = self.get_protocol_comparison_data(experiment_id, use_max_accuracy=use_max_accuracy)
        if not comparison_data:
            return pd.DataFrame()
            
        # Collect all metric values per protocol
        protocol_metrics = {protocol: [] for protocol in comparison_data['protocols']}
        
        for attack_type in comparison_data['attack_types']:
            if attack_type not in comparison_data['comparison_data']:
                continue
                
            attack_data = comparison_data['comparison_data'][attack_type]
            if metric not in attack_data:
                continue
                
            for bias in comparison_data['bias_values']:
                if bias not in attack_data[metric]:
                    continue
                    
                bias_data = attack_data[metric][bias]
                if not bias_data:
                    continue
                    
                for protocol, score in bias_data.items():
                    protocol_metrics[protocol].append(score)
        
        # Calculate statistics for each protocol
        ranking_data = []
        
        for protocol, scores in protocol_metrics.items():
            if not scores:
                continue
                
            ranking_data.append({
                'protocol': protocol,
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'std_dev': np.std(scores) if len(scores) > 1 else 0,
                'data_points': len(scores)
            })
        
        # Create DataFrame and sort by average score
        ranking_df = pd.DataFrame(ranking_data)
        if not ranking_df.empty:
            ranking_df = ranking_df.sort_values(
                by='avg_score', 
                ascending=not is_higher_better
            ).reset_index(drop=True)
            
            # Add rank column
            ranking_df['rank'] = ranking_df.index + 1
            
            # Reorder columns
            ranking_df = ranking_df[['rank', 'protocol', 'avg_score', 'min_score', 'max_score', 'std_dev', 'data_points']]
        
        return ranking_df
    
    def compare_hierarchical_fl(self, experiment_id=None, metric='round_accuracy', is_higher_better=True, use_max_accuracy=False):
        """
        Compare hierarchical FL performance against other protocols

        Args:
            experiment_id (int, optional): Experiment ID to filter by
            metric (str): Metric to evaluate (e.g., 'round_accuracy', 'round_backdoor_success')
            is_higher_better (bool): Whether higher values are better (True for accuracy, False for backdoor)
            use_max_accuracy (bool, optional): Whether to use the maximum accuracy across all rounds
                                               instead of the final round accuracy. Defaults to False.

        Returns:
            dict: Dictionary with comparison statistics
        """
        comparison_data = self.get_protocol_comparison_data(experiment_id, use_max_accuracy=use_max_accuracy)
        if not comparison_data:
            return {}
            
        # Find hierarchical FL protocol name
        hier_protocol = None
        for protocol in comparison_data['protocols']:
            if 'hierarchical' in protocol.lower() or 'heirichal' in protocol.lower():
                hier_protocol = protocol
                break
                
        if not hier_protocol:
            return {'error': 'Hierarchical FL protocol not found'}
        
        comparison_stats = {
            'protocol': hier_protocol,
            'win_count': 0,
            'loss_count': 0,
            'tie_count': 0,
            'win_percentage': 0,
            'average_difference': 0,
            'details': {}
        }
        
        total_comparisons = 0
        total_difference = 0
        
        for attack_type in comparison_data['attack_types']:
            if attack_type not in comparison_data['comparison_data']:
                continue
                
            attack_data = comparison_data['comparison_data'][attack_type]
            if metric not in attack_data:
                continue
                
            comparison_stats['details'][attack_type] = {}
            
            for bias in comparison_data['bias_values']:
                if bias not in attack_data[metric]:
                    continue
                    
                bias_data = attack_data[metric][bias]
                if not bias_data or hier_protocol not in bias_data:
                    continue
                    
                hier_score = bias_data[hier_protocol]
                
                # Find best non-hierarchical score
                best_other_protocol = None
                best_other_score = None
                
                for protocol, score in bias_data.items():
                    if protocol == hier_protocol:
                        continue
                        
                    if best_other_score is None or (is_higher_better and score > best_other_score) or (not is_higher_better and score < best_other_score):
                        best_other_score = score
                        best_other_protocol = protocol
                
                if best_other_protocol is None:
                    continue
                    
                # Compare scores
                difference = hier_score - best_other_score
                
                # For backdoor success, reverse the interpretation (lower is better)
                if not is_higher_better:
                    difference = -difference
                    
                comparison_stats['details'][attack_type][bias] = {
                    'hier_score': hier_score,
                    'best_other_protocol': best_other_protocol,
                    'best_other_score': best_other_score,
                    'difference': difference,
                    'result': 'win' if difference > 0 else ('tie' if difference == 0 else 'loss')
                }
                
                # Update summary statistics
                if difference > 0:
                    comparison_stats['win_count'] += 1
                elif difference < 0:
                    comparison_stats['loss_count'] += 1
                else:
                    comparison_stats['tie_count'] += 1
                    
                total_comparisons += 1
                total_difference += difference
        
        # Calculate summary statistics
        if total_comparisons > 0:
            comparison_stats['win_percentage'] = comparison_stats['win_count'] / total_comparisons * 100
            comparison_stats['average_difference'] = total_difference / total_comparisons
        
        return comparison_stats
    
    def get_performance_timeline(self, experiment_id=None, protocol=None, attack_type=None, bias_value=None, include_max_accuracy=False):
        """
        Get performance metrics over time for specified configuration

        Args:
            experiment_id (int, optional): Experiment ID to filter by
            protocol (str, optional): Filter by specific protocol (None for all)
            attack_type (str, optional): Filter by attack type (None for all)
            bias_value (float, optional): Filter by bias value (None for all)
            include_max_accuracy (bool, optional): Whether to include a column showing the maximum accuracy
                                                   achieved up to each round. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with performance metrics by round
        """
        with self._lock:
            # Get experiment results
            results = self.get_experiment_results(experiment_id)
            
            if results is None or results.empty:
                return pd.DataFrame()
                
            # Apply filters
            filtered_results = results.copy()
            
            if protocol is not None and 'aggregation_name' in filtered_results.columns:
                filtered_results = filtered_results[filtered_results['aggregation_name'] == protocol]
                
            if attack_type is not None:
                if attack_type == 'No Attack':
                    filtered_results = filtered_results[filtered_results['malicious_type'].isnull()]
                elif 'malicious_type' in filtered_results.columns:
                    filtered_results = filtered_results[filtered_results['malicious_type'] == attack_type]
                    
            if bias_value is not None and 'bias_values' in filtered_results.columns:
                filtered_results = filtered_results[filtered_results['bias_values'] == bias_value]
                
            if filtered_results.empty:
                return pd.DataFrame()
                
            # Determine round column
            round_col = 'round_num' if 'round_num' in filtered_results.columns else 'round'
            
            # Group by round and calculate metrics
            metrics = ['round_accuracy']
            if 'round_backdoor_success' in filtered_results.columns:
                metrics.append('round_backdoor_success')
                
            # Select relevant columns
            group_cols = [round_col]
            if protocol is None and 'aggregation_name' in filtered_results.columns:
                group_cols.append('aggregation_name')
                
            if attack_type is None and 'malicious_type' in filtered_results.columns:
                group_cols.append('malicious_type')
                
            if bias_value is None and 'bias_values' in filtered_results.columns:
                group_cols.append('bias_values')
                
            # Calculate aggregated metrics
            timeline_data = filtered_results.groupby(group_cols)[metrics].mean().reset_index()

            # If requested, add a column showing max accuracy achieved up to each round
            if include_max_accuracy and 'round_accuracy' in timeline_data.columns:
                # Process for each protocol separately (if applicable)
                if 'aggregation_name' in timeline_data.columns:
                    for protocol in timeline_data['aggregation_name'].unique():
                        protocol_mask = timeline_data['aggregation_name'] == protocol
                        protocol_data = timeline_data[protocol_mask]

                        # Calculate cumulative max accuracy for each protocol
                        if round_col in protocol_data.columns:
                            # Sort by round
                            protocol_data = protocol_data.sort_values(by=round_col)
                            # Calculate cumulative max
                            timeline_data.loc[protocol_mask, 'max_accuracy_to_date'] = protocol_data['round_accuracy'].cummax()
                else:
                    # If no protocol distinction, simply calculate for the entire dataset
                    if round_col in timeline_data.columns:
                        # Sort by round
                        timeline_data = timeline_data.sort_values(by=round_col)
                        # Calculate cumulative max
                        timeline_data['max_accuracy_to_date'] = timeline_data['round_accuracy'].cummax()

            return timeline_data