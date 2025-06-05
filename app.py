import streamlit as st
import os
import time
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from data_loader import DataLoader
from file_watcher import FileWatcher
from sidebar_template import create_sidebar

# Set page configuration
st.set_page_config(
    page_title="Hierarchical Federated Learning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'data_loader' not in st.session_state:
    # Set the data directory - modify this path as needed
    data_dir = Path("M:/PythonTests/newSafeFL/SAFEFL/results/hierarchical")
    st.session_state.data_loader = DataLoader(data_dir)
    st.session_state.file_watcher = FileWatcher(data_dir)

# Create consistent sidebar
create_sidebar()

# Main content
st.title("Hierarchical Federated Learning Dashboard")

# Get current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.subheader(f"Experiment ID: {current_exp_id}")

# Overview metrics
st.header("Experiment Overview")

# Get experiment overview data
experiment_data = st.session_state.data_loader.get_experiment_results()

if experiment_data is not None and not experiment_data.empty:
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Accuracy", f"{experiment_data['round_accuracy'].iloc[0]:.2%}")
    
    with col2:
        if 'round_backdoor_success' in experiment_data.columns and not pd.isna(experiment_data['round_backdoor_success'].iloc[0]):
            st.metric("Backdoor Success", f"{experiment_data['round_backdoor_success'].iloc[0]:.2%}")
        else:
            st.metric("Backdoor Success", "N/A")
    
    with col3:
        st.metric("Total Participants", f"{experiment_data['total_participants'].iloc[0]}")
    
    with col4:
        st.metric("Malicious Participants", f"{experiment_data['malicious_count'].iloc[0]}")
        
    # Additional experiment information
    st.subheader("Experiment Configuration")

    # Display separate experiment details when using different experiment IDs
    if st.session_state.data_loader.use_separate_hierarchical:
        st.warning("**Using separate experiment IDs for HierarchicalFL and other aggregation rules**")

        # Get the experiment details for both experiment IDs
        main_exp_id = st.session_state.data_loader.current_experiment_id
        hier_exp_id = st.session_state.data_loader.hierarchical_experiment_id

        # Get data for main experiment
        main_exp_data = st.session_state.data_loader.get_experiment_results(main_exp_id)

        # Get data for hierarchical experiment (filtering for hierarchical FL only)
        hier_exp_full = st.session_state.data_loader.get_experiment_results(hier_exp_id)
        hier_exp_data = None

        if hier_exp_full is not None and not hier_exp_full.empty:
            # Filter for hierarchical FL data
            hierarchical_mask = (hier_exp_full['aggregation_name'].str.lower().str.contains('hierarchical') |
                                hier_exp_full['aggregation_name'].str.lower().str.contains('heirichal'))
            hier_exp_data = hier_exp_full[hierarchical_mask]

        # Create expandable sections for detailed experiment parameters
        with st.expander("Main Experiment Parameters (Other Aggregation Rules)", expanded=False):
            if main_exp_data is not None and not main_exp_data.empty:
                # Get a representative row for the main experiment (any non-hierarchical)
                non_hier_mask = ~(main_exp_data['aggregation_name'].str.lower().str.contains('hierarchical') |
                                main_exp_data['aggregation_name'].str.lower().str.contains('heirichal'))
                non_hier_data = main_exp_data[non_hier_mask]

                if not non_hier_data.empty:
                    st.info(f"""
                    - **Experiment ID**: {main_exp_id}
                    - **Attack Type**: {non_hier_data['malicious_type'].iloc[0]}
                    - **Number of Workers**: {non_hier_data['total_participants'].iloc[0]}
                    - **Malicious Workers**: {non_hier_data['malicious_count'].iloc[0]}
                    - **Bias Values**: {non_hier_data['bias_values'].iloc[0]}
                    - **Server Bias**: {non_hier_data['server_bias'].iloc[0]}
                    - **Completed Rounds**: {non_hier_data['round_num'].iloc[0]}
                    """)
                else:
                    st.info(f"Experiment ID: {main_exp_id} - No non-hierarchical aggregation data found")
            else:
                st.info(f"Experiment ID: {main_exp_id} - No data found")

        with st.expander("HierarchicalFL Experiment Parameters", expanded=False):
            if hier_exp_data is not None and not hier_exp_data.empty:
                st.info(f"""
                - **Experiment ID**: {hier_exp_id}
                - **Aggregation Method**: {hier_exp_data['aggregation_name'].iloc[0]}
                - **Attack Type**: {hier_exp_data['malicious_type'].iloc[0]}
                - **Number of Workers**: {hier_exp_data['total_participants'].iloc[0]}
                - **Malicious Workers**: {hier_exp_data['malicious_count'].iloc[0]}
                - **Bias Values**: {hier_exp_data['bias_values'].iloc[0]}
                - **Server Bias**: {hier_exp_data['server_bias'].iloc[0]}
                - **Completed Rounds**: {hier_exp_data['round_num'].iloc[0]}
                """)

                # Try to extract HierarchicalFL specific parameters

                # Check if detailed parameters CSV exists
                hier_exp_id_str = str(hier_exp_id)
                params_file = Path(f"results/hierarchical/agg_parameters_{hier_exp_id_str}.csv")

                if params_file.exists():
                    try:
                        import json
                        params_df = pd.read_csv(params_file)
                        # Find HierarchicalFL specific parameters
                        hfl_params_row = params_df[params_df['aggregation_name'] == 'heirichalFL']

                        if not hfl_params_row.empty:
                            # Parse the JSON parameters
                            params_json = json.loads(hfl_params_row['parameters'].iloc[0])

                            # Display the parameters
                            params_text = "**HierarchicalFL Command-line Parameters**:\n"
                            for param, value in params_json.items():
                                params_text += f"- **{param}**: {value}\n"

                            st.info(params_text)
                    except Exception as e:
                        st.warning(f"Error loading HierarchicalFL parameters: {str(e)}")

                # Fallback to experiment config extraction if available
                elif 'experiment_config' in hier_exp_data.columns:
                    hfl_params = hier_exp_data['experiment_config'].iloc[0].split('_')
                    n_groups = next((p for p in hfl_params if p.startswith('n_groups=')), None)
                    agg_rule = next((p for p in hfl_params if p.startswith('agg_rule=')), None)
                    assumed_mal = next((p for p in hfl_params if p.startswith('assumed_mal_prct=')), None)

                    if any([n_groups, agg_rule, assumed_mal]):
                        st.info(f"""
                        **HierarchicalFL Specific Parameters**:
                        {f"- **Number of Groups**: {n_groups.split('=')[1]}" if n_groups else ""}
                        {f"- **Group Agg Rule**: {agg_rule.split('=')[1]}" if agg_rule else ""}
                        {f"- **Assumed Malicious %**: {assumed_mal.split('=')[1]}" if assumed_mal else ""}
                        """)
            else:
                st.info(f"Experiment ID: {hier_exp_id} - No HierarchicalFL data found")

    # Default experiment details display
    # Create two columns for experiment details
    config_col1, config_col2 = st.columns(2)

    with config_col1:
        st.info(f"""
        - **Aggregation Method**: {experiment_data['aggregation_name'].iloc[0]}
        - **Attack Type**: {experiment_data['malicious_type'].iloc[0]}
        - **Completed Rounds**: {experiment_data['round_num'].iloc[0]}
        """)

    with config_col2:
        st.info(f"""
        - **Server Bias**: {experiment_data['server_bias'].iloc[0]}
        - **Bias Values**: {experiment_data['bias_values'].iloc[0]}
        - **Experiment ID**: {experiment_data['experiment_id'].iloc[0]}
        """)
else:
    st.warning("Experiment results data is not available. Please check the data directory.")

# Add option to use max accuracy instead of final accuracy
use_max_accuracy = st.checkbox("Use Maximum Accuracy (instead of final round)", value=False,
                             help="When enabled, visualizations will use the maximum accuracy achieved during training instead of the accuracy from the final round")

# Get all experiment data for comparison
all_experiment_data = st.session_state.data_loader.get_experiment_results(None)

# Store the accuracy mode preference in session state
if 'use_max_accuracy' not in st.session_state:
    st.session_state.use_max_accuracy = use_max_accuracy
elif st.session_state.use_max_accuracy != use_max_accuracy:
    st.session_state.use_max_accuracy = use_max_accuracy
    st.experimental_rerun()

# Focus on different aggregation rules and their performance under the specified attack types
if all_experiment_data is not None and not all_experiment_data.empty:
    # Define the attacks we want to focus on - exactly as specified
    attack_types_of_interest = ["no", "label_flipping_attack", "scaling_attack_insert_backdoor", "scaling_attack_scale"]
    
    # Filter the dataframe to get only the attacks of interest
    # Handle "no" attack which might be represented as NaN or "no-byz"
    filtered_data = all_experiment_data[
        (all_experiment_data['malicious_type'].isin(attack_types_of_interest)) | 
        (all_experiment_data['malicious_type'].isna() & all_experiment_data['malicious_count'] == 0) |
        (all_experiment_data['malicious_type'] == "no-byz")
    ].copy()
    
    # Replace NaN/no-byz malicious_type with "no" for better labeling consistency
    filtered_data.loc[filtered_data['malicious_type'].isna(), 'malicious_type'] = "no"
    filtered_data.loc[filtered_data['malicious_type'] == "no-byz", 'malicious_type'] = "no"
    
    # Get unique aggregation methods and actual available attack types
    aggregation_methods = filtered_data['aggregation_name'].unique()
    available_attack_types = filtered_data['malicious_type'].unique()
    
    # Performance Comparison Section
    st.header("Aggregation Rule Performance Under Different Attacks")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Comparison Tables", "Bar Charts", "Line Charts"])
    
    with tab1:
        st.subheader("Performance Comparison by Aggregation Method and Attack Type")
        
        try:
            # Get protocol comparison data with the selected accuracy mode
            accuracy_mode_text = "maximum" if st.session_state.use_max_accuracy else "final round"
            st.info(f"Showing {accuracy_mode_text} accuracy values for all visualizations and comparisons.")

            # Group by aggregation method and attack type with error handling
            comparison_df = filtered_data.groupby(['aggregation_name', 'malicious_type']).agg({
                'round_accuracy': ['mean', 'min', 'max', 'std'],
                'round_backdoor_success': ['mean', 'min', 'max', 'std'],
            }).reset_index()
            
            # Flatten multi-index columns
            comparison_df.columns = ['_'.join(col).strip('_') for col in comparison_df.columns.values]
            
            # Create a nice comparison table
            table_data = []
            for agg in aggregation_methods:
                for attack in available_attack_types:
                    row_data = comparison_df[
                        (comparison_df['aggregation_name'] == agg) & 
                        (comparison_df['malicious_type'] == attack)
                    ]
                    
                    if not row_data.empty:
                        table_row = {
                            'Aggregation': agg,
                            'Attack': attack,
                            'Mean Accuracy': f"{row_data['round_accuracy_mean'].iloc[0]:.2%}",
                            'Min Accuracy': f"{row_data['round_accuracy_min'].iloc[0]:.2%}",
                            'Max Accuracy': f"{row_data['round_accuracy_max'].iloc[0]:.2%}",
                        }
                        
                        # Add backdoor success if available (only for scaling attacks)
                        if 'round_backdoor_success_mean' in row_data.columns and not pd.isna(row_data['round_backdoor_success_mean'].iloc[0]):
                            table_row['Backdoor Success'] = f"{row_data['round_backdoor_success_mean'].iloc[0]:.2%}"
                        else:
                            table_row['Backdoor Success'] = "N/A"
                            
                        table_data.append(table_row)
            
            # Display the comparison table
            if table_data:
                comparison_table = pd.DataFrame(table_data)
                st.dataframe(comparison_table, use_container_width=True)
            else:
                st.warning("No comparison data available for the specified attack types.")
        except Exception as e:
            st.error(f"Error generating comparison table: {str(e)}")
    
    with tab2:
        st.subheader("Accuracy Comparison by Aggregation Method and Attack Type")
        
        try:
            # Create data for the chart
            chart_data = []
            for agg in aggregation_methods:
                for attack in available_attack_types:
                    row_data = filtered_data[
                        (filtered_data['aggregation_name'] == agg) & 
                        (filtered_data['malicious_type'] == attack)
                    ]
                    
                    if not row_data.empty:
                        backdoor_value = np.nan
                        if 'round_backdoor_success' in row_data.columns:
                            backdoor_value = row_data['round_backdoor_success'].mean()
                            
                        chart_data.append({
                            'Aggregation': agg,
                            'Attack': attack,
                            'Accuracy': row_data['round_accuracy'].mean(),
                            'Backdoor': backdoor_value
                        })
            
            if chart_data:
                chart_df = pd.DataFrame(chart_data)

                # Add note about using max/final accuracy
                accuracy_mode = "maximum" if st.session_state.use_max_accuracy else "final round"
                st.caption(f"Displaying {accuracy_mode} accuracy values for all charts")

                # Create bar chart comparing accuracy for each attack type across aggregation methods
                accuracy_by_attack = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X('Aggregation:N', title='Aggregation Method'),
                    y=alt.Y('Accuracy:Q', title='Mean Accuracy', scale=alt.Scale(domain=[0, 1])),
                    color='Aggregation:N',
                    column='Attack:N',
                    tooltip=['Aggregation', 'Attack', alt.Tooltip('Accuracy:Q', format='.2%')]
                ).properties(
                    width=200,
                    title='Mean Accuracy by Aggregation Method and Attack Type'
                )
                
                st.altair_chart(accuracy_by_attack, use_container_width=True)
                
                # Alternate view - Group by aggregation method instead of attack type
                st.subheader("Accuracy by Attack Type for Each Aggregation Method")
                
                accuracy_by_agg = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X('Attack:N', title='Attack Type'),
                    y=alt.Y('Accuracy:Q', title='Mean Accuracy', scale=alt.Scale(domain=[0, 1])),
                    color='Attack:N',
                    column='Aggregation:N',
                    tooltip=['Aggregation', 'Attack', alt.Tooltip('Accuracy:Q', format='.2%')]
                ).properties(
                    width=150,
                    title='Mean Accuracy by Attack Type for Each Aggregation Method'
                )
                
                st.altair_chart(accuracy_by_agg, use_container_width=True)
                
                # Create a separate chart for backdoor success rate (filtering out NaN values)
                backdoor_df = chart_df.dropna(subset=['Backdoor'])
                
                if not backdoor_df.empty:
                    st.subheader("Backdoor Success Rate by Aggregation Method")
                    
                    # For backdoor success, focus on the attacks that have backdoor success metrics
                    backdoor_attacks = backdoor_df['Attack'].unique()
                    
                    # Create a chart for each attack that has backdoor success metrics
                    for attack in backdoor_attacks:
                        attack_data = backdoor_df[backdoor_df['Attack'] == attack]
                        
                        backdoor_chart = alt.Chart(attack_data).mark_bar().encode(
                            x=alt.X('Aggregation:N', title='Aggregation Method'),
                            y=alt.Y('Backdoor:Q', title='Backdoor Success Rate', scale=alt.Scale(domain=[0, 1])),
                            color='Aggregation:N',
                            tooltip=['Aggregation', 'Attack', alt.Tooltip('Backdoor:Q', format='.2%')]
                        ).properties(
                            width=600,
                            height=300,
                            title=f'Backdoor Success Rate by Aggregation Method for {attack}'
                        )
                        
                        st.altair_chart(backdoor_chart, use_container_width=True)
            else:
                st.warning("No chart data available for the specified attack types.")
        except Exception as e:
            st.error(f"Error generating bar charts: {str(e)}")
    
    with tab3:
        st.subheader("Detailed Performance Comparison")
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create a selection for attack type from available attacks
            selected_attack = st.selectbox(
                "Select Attack Type:",
                options=available_attack_types,
                format_func=lambda x: "No Attack" if x == "no" else x
            )
            
            # Create radio buttons to select the visualization type
            chart_type = st.radio(
                "Select Chart Type:",
                ["Accuracy Comparison", "Backdoor Success", "Combined View"]
            )
            
            # Add a checkbox to show standard deviation
            show_std = st.checkbox("Show Standard Deviation", value=True)
        
        with col2:
            try:
                # Filter data for the selected attack
                attack_data = filtered_data[filtered_data['malicious_type'] == selected_attack].copy()
                
                if not attack_data.empty:
                    # Create pivot table with performance metrics by aggregation method
                    agg_dict = {'round_accuracy': ['mean', 'std']}
                    if 'round_backdoor_success' in attack_data.columns:
                        agg_dict['round_backdoor_success'] = ['mean', 'std']
                        
                    pivot_data = attack_data.pivot_table(
                        index='aggregation_name', 
                        values=list(agg_dict.keys()),
                        aggfunc=list(set(sum(agg_dict.values(), [])))  # Flatten the list of aggregation functions
                    ).reset_index()
                    
                    # Flatten multi-index columns 
                    pivot_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in pivot_data.columns]
                    
                    # Check if required columns exist
                    has_accuracy_mean = 'mean_round_accuracy' in pivot_data.columns
                    has_accuracy_std = 'std_round_accuracy' in pivot_data.columns
                    has_backdoor_mean = 'mean_round_backdoor_success' in pivot_data.columns
                    has_backdoor_std = 'std_round_backdoor_success' in pivot_data.columns
                    
                    # Depending on selected chart type, show different visualizations
                    if chart_type == "Accuracy Comparison" and has_accuracy_mean:
                        # Prepare data for the chart
                        acc_data = pd.DataFrame({
                            'Aggregation': pivot_data['aggregation_name'],
                            'Accuracy': pivot_data['mean_round_accuracy'],
                            'Error': pivot_data['std_round_accuracy'] if has_accuracy_std and show_std else 0
                        })
                        
                        # Create bar chart with optional error bars
                        accuracy_base = alt.Chart(acc_data).encode(
                            x=alt.X('Aggregation:N', title='Aggregation Method'),
                            y=alt.Y('Accuracy:Q', title='Model Accuracy', scale=alt.Scale(domain=[0, 1])),
                            tooltip=['Aggregation', alt.Tooltip('Accuracy:Q', format='.2%')]
                        )
                        
                        # Create bars
                        bars = accuracy_base.mark_bar(color='steelblue')
                        
                        # Add error bars if requested
                        if show_std and has_accuracy_std:
                            error_bars = accuracy_base.mark_errorbar(color='black').encode(
                                y='Accuracy-Error:Q',
                                y2='Accuracy+Error:Q'
                            )
                            accuracy_chart = (bars + error_bars).properties(
                                width=500,
                                height=400,
                                title=f'Model Accuracy by Aggregation Method for {selected_attack}'
                            )
                        else:
                            accuracy_chart = bars.properties(
                                width=500,
                                height=400,
                                title=f'Model Accuracy by Aggregation Method for {selected_attack}'
                            )
                        
                        st.altair_chart(accuracy_chart, use_container_width=True)
                    
                    elif chart_type == "Backdoor Success" and has_backdoor_mean:
                        # Filter out NaN values
                        backdoor_data = pd.DataFrame({
                            'Aggregation': pivot_data['aggregation_name'],
                            'Backdoor': pivot_data['mean_round_backdoor_success'],
                            'Error': pivot_data['std_round_backdoor_success'] if has_backdoor_std and show_std else 0
                        }).dropna(subset=['Backdoor'])
                        
                        if not backdoor_data.empty:
                            # Create bar chart with optional error bars
                            backdoor_base = alt.Chart(backdoor_data).encode(
                                x=alt.X('Aggregation:N', title='Aggregation Method'),
                                y=alt.Y('Backdoor:Q', title='Backdoor Success Rate', scale=alt.Scale(domain=[0, 1])),
                                tooltip=['Aggregation', alt.Tooltip('Backdoor:Q', format='.2%')]
                            )
                            
                            # Create bars
                            bars = backdoor_base.mark_bar(color='firebrick')
                            
                            # Add error bars if requested
                            if show_std and has_backdoor_std:
                                error_bars = backdoor_base.mark_errorbar(color='black').encode(
                                    y='Backdoor-Error:Q',
                                    y2='Backdoor+Error:Q'
                                )
                                backdoor_chart = (bars + error_bars).properties(
                                    width=500,
                                    height=400,
                                    title=f'Backdoor Success Rate by Aggregation Method for {selected_attack}'
                                )
                            else:
                                backdoor_chart = bars.properties(
                                    width=500,
                                    height=400,
                                    title=f'Backdoor Success Rate by Aggregation Method for {selected_attack}'
                                )
                            
                            st.altair_chart(backdoor_chart, use_container_width=True)
                        else:
                            st.info("No backdoor success data available for this attack type.")
                    elif chart_type == "Backdoor Success" and not has_backdoor_mean:
                        st.info("Backdoor success metrics are not available for this attack type.")
                    
                    elif chart_type == "Combined View" and has_accuracy_mean:
                        # Check if both accuracy and backdoor success data exist
                        has_backdoor = has_backdoor_mean and not pivot_data['mean_round_backdoor_success'].isna().all()
                        
                        # Create a matplotlib plot for the combined view
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        
                        # Plot accuracy bars
                        x = np.arange(len(pivot_data))
                        bar_width = 0.35
                        
                        ax1.set_xlabel('Aggregation Method')
                        ax1.set_ylabel('Model Accuracy', color='tab:blue')
                        bars1 = ax1.bar(x, pivot_data['mean_round_accuracy'], bar_width, label='Accuracy', color='tab:blue', alpha=0.7)
                        ax1.tick_params(axis='y', labelcolor='tab:blue')
                        ax1.set_ylim(0, 1)
                        
                        # Add error bars if requested
                        if show_std and has_accuracy_std:
                            ax1.errorbar(x, pivot_data['mean_round_accuracy'], yerr=pivot_data['std_round_accuracy'], 
                                        fmt='none', color='black', capsize=5)
                        
                        # Plot backdoor success if available
                        if has_backdoor:
                            ax2 = ax1.twinx()
                            ax2.set_ylabel('Backdoor Success Rate', color='tab:red')
                            
                            # Filter out NaN values for backdoor success
                            valid_idx = ~pivot_data['mean_round_backdoor_success'].isna()
                            
                            if any(valid_idx):
                                filtered_x = x[valid_idx.values]  # Convert to numpy array
                                filtered_backdoor = pivot_data.loc[valid_idx, 'mean_round_backdoor_success'].values
                                
                                # Only get std if it exists and show_std is True
                                filtered_std = None
                                if show_std and has_backdoor_std:
                                    filtered_std = pivot_data.loc[valid_idx, 'std_round_backdoor_success'].values
                                
                                bars2 = ax2.bar(filtered_x + bar_width, filtered_backdoor, bar_width, 
                                              label='Backdoor Success', color='tab:red', alpha=0.7)
                                
                                # Add error bars if requested
                                if show_std and filtered_std is not None:
                                    ax2.errorbar(filtered_x + bar_width, filtered_backdoor, yerr=filtered_std, 
                                               fmt='none', color='black', capsize=5)
                                
                                ax2.tick_params(axis='y', labelcolor='tab:red')
                                ax2.set_ylim(0, 1)
                                
                        # Set x-axis ticks to aggregation names
                        ax1.set_xticks(x + bar_width / 2 if has_backdoor else x)
                        ax1.set_xticklabels(pivot_data['aggregation_name'], rotation=45, ha='right')
                        
                        # Add legend
                        lines, labels = ax1.get_legend_handles_labels()
                        if has_backdoor:
                            lines2, labels2 = ax2.get_legend_handles_labels()
                            ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
                        else:
                            ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
                        
                        plt.title(f'Performance Metrics for {selected_attack}')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    
                    # Display summary statistics table
                    st.subheader("Summary Statistics")
                    
                    # Format for display
                    display_columns = {
                        'Aggregation': pivot_data['aggregation_name']
                    }
                    
                    if has_accuracy_mean:
                        display_columns['Mean Accuracy'] = pivot_data['mean_round_accuracy'].map('{:.2%}'.format)
                    
                    if has_accuracy_std:
                        display_columns['Std Dev (Accuracy)'] = pivot_data['std_round_accuracy'].map('{:.4f}'.format)
                    
                    # Add backdoor success if available
                    if has_backdoor_mean:
                        display_columns['Mean Backdoor Success'] = pivot_data['mean_round_backdoor_success'].map(
                            lambda x: '{:.2%}'.format(x) if not pd.isna(x) else "N/A"
                        )
                    
                    if has_backdoor_std:
                        display_columns['Std Dev (Backdoor)'] = pivot_data['std_round_backdoor_success'].map(
                            lambda x: '{:.4f}'.format(x) if not pd.isna(x) else "N/A"
                        )
                    
                    display_data = pd.DataFrame(display_columns)
                    st.dataframe(display_data, use_container_width=True)
                else:
                    st.warning(f"No data available for {selected_attack}.")
            except Exception as e:
                st.error(f"Error in detailed comparison: {str(e)}")
                st.error("Detailed error info:")
                st.exception(e)
else:
    st.warning("No experiment data available for comparison.")

# Navigation to other pages
st.header("Dashboard Navigation")
st.markdown("""
Explore detailed visualizations in the following pages:

1. **Model Performance**: Track model accuracy and backdoor attack success over rounds
2. **Detailed Model Performance**: Compare performance by specific configuration combinations
3. **Group Analysis**: Visualize group scores and malicious user distribution
4. **User Scores**: Explore individual user scores and filtering effectiveness
5. **Detailed User Scores**: Analyze user scores for specific configurations
6. **Group Scores**: Visualize group formation and malicious user distribution
7. **Detailed Group Scores**: Analyze group scores for specific configurations
""")

# Pass the max accuracy setting to session state so other pages can use it
st.session_state.use_max_accuracy = use_max_accuracy

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | Real-time Monitoring System")