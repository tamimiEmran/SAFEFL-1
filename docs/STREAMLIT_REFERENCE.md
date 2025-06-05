# SafeFL Streamlit Dashboard Reference

## Application Structure Overview

The Streamlit dashboard is designed to visualize and analyze the results of hierarchical federated learning experiments, with a focus on comparing different aggregation methods under various attack scenarios.

### Main Components

1. **Main Dashboard (`app.py`)**: 
   - Entry point for the Streamlit application
   - Displays experiment overview and aggregation rule comparisons
   - Provides navigation to other pages

2. **Data Loading (`data_loader.py`)**:
   - Core class that loads and manages experiment data
   - Provides methods to filter and retrieve data by experiment ID, round, etc.
   - Handles caching and thread safety for data access

3. **File Monitoring (`file_watcher.py`)**:
   - Monitors the results directory for changes
   - Triggers data reload when files are updated

4. **Sidebar Template (`sidebar_template.py`)**:
   - Provides consistent sidebar across all pages
   - Handles experiment selection and data refresh

5. **Pages**:
   - **Model Performance (`1_Model_Performance.py`)**: Compares aggregation methods' performance
   - **User Scores (`4_User_Scores.py`)**: Analyzes individual user scores and filtering
   - **Group Scores (`6_Group_Scores.py`)**: Visualizes group formation and malicious user distribution

## Data Structure and CSV Files

The dashboard relies on several CSV files stored in the `results/hierarchical` directory:

| File Name | Description | Key Columns |
|-----------|-------------|-------------|
| `experiment_results.csv` | Final results for each experiment configuration | `round_num`, `round_accuracy`, `round_backdoor_success`, `aggregation_name`, `malicious_count`, `malicious_type`, `bias_values`, `experiment_id` |
| `group_scores.csv` | Scores assigned to each group per round | `round`, `group_id`, `score`, `actual_malicious_count`, `filtered_count`, `experiment_id`, `attack_type` |
| `user_scores.csv` | Scores assigned to individual users | `round`, `user_id`, `score`, `adjustment`, `is_actually_malicious`, `is_filtered_out`, `experiment_id`, `attack_type` |
| `user_membership.csv` | Group assignment for each user | `round`, `user_id`, `group_id`, `is_actually_malicious`, `is_filtered_out`, `experiment_id`, `attack_type` |
| `global_gradients.csv` | Gradient values after aggregation | `round`, `gradient_norm`, `filtered_users_count`, `experiment_id`, `attack_type`, `component_0...N` |
| `summary_stats.csv` | Statistical metrics per round | Various statistics including `precision`, `recall`, `f1_score`, etc. |

## Data Flow and Loading Process

1. **Initialization**:
   - The `app.py` initializes the `DataLoader` with the data directory path
   - Initial data is loaded from all CSV files into memory caches
   - A `FileWatcher` instance is created to monitor for data changes

2. **Data Fetching**:
   - Components request data via `data_loader` methods (e.g., `get_experiment_results()`)
   - Data is filtered based on the currently selected experiment ID
   - Methods provide thread-safe access to the data via an internal lock

3. **Data Refresh**:
   - User can manually refresh data using a button in the sidebar
   - Automatic refresh occurs at intervals set by the user
   - When files change, `file_watcher` detects changes and triggers a reload

4. **Page Rendering**:
   - Each page accesses filtered data relevant to its visualizations
   - Filtering can be done by experiment ID, attack type, round number, etc.
   - All pages share the same `data_loader` instance via Streamlit's session state

## Current Issues and Improvement Areas

1. **Page Organization**:
   - Missing pages 2, 3, and 5 in the page numbering sequence
   - Inconsistent page navigation and references

2. **Data Filtering**:
   - Filtering logic is duplicated across multiple pages
   - Some filtering options in sidebar may overwrite other pages' selections

3. **Error Handling**:
   - Limited error handling for missing data or incorrectly formatted CSV files
   - Experiement switching can lead to errors when data schemas differ

4. **Performance Issues**:
   - Large CSV files may cause memory usage issues
   - No pagination for large datasets in tables
   - Repeated data loading and filtering operations

5. **UI/UX Improvements**:
   - Inconsistent layout between pages
   - Limited explanatory text for complex visualizations
   - Some visualization options are hidden in expanders or tabs

6. **Code Organization**:
   - Common visualization functions are duplicated rather than shared
   - Hardcoded paths and values in several locations
   - Some visualizations have hardcoded column names that might not be present

## Future Improvement Suggestions

1. **Architecture Improvements**:
   - Create a unified filtering component shared across pages
   - Implement proper route handling for page navigation
   - Add data validation to ensure CSV files meet expected schema

2. **Additional Features**:
   - Add export functionality for charts and tables
   - Implement a configuration page for setting global parameters
   - Add comparative analysis between multiple experiments
   - Create custom visualization for hierarchical structure

3. **Performance Optimizations**:
   - Implement data sampling for large datasets
   - Add pagination for tables
   - Optimize data loading with incremental updates

4. **UI/UX Enhancements**:
   - Standardize layout and styling across pages
   - Add tooltips and explanatory text for complex metrics
   - Improve filtering UI with more intuitive controls
   - Add progress indicators for long-running operations

5. **Documentation**:
   - Add inline documentation for complex visualizations
   - Create a help section explaining metrics and calculations
   - Document the data schema and relationships between files