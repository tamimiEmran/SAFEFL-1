import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sidebar_template import create_sidebar

# Set page configuration
st.set_page_config(
    page_title="2. Detailed Model Performance - Hierarchical Federated Learning",
    page_icon="📊",
    layout="wide"
)

# Ensure data loader is available
if 'data_loader' not in st.session_state:
    st.error("Please navigate to the main page first to initialize the data loader.")
    st.stop()

# Page title
st.title("2. Detailed Model Performance")
st.markdown("Compare performance by combination of malicious count, attack type, and bias value")

# Display current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.info(f"Currently viewing Experiment ID: {current_exp_id}")

# Create consistent sidebar
create_sidebar()

# Get experiment results
experiment_results = st.session_state.data_loader.get_experiment_results()

if experiment_results is not None and not experiment_results.empty:
    # Create a section for the new detailed visualization
    st.header("Detailed Performance Comparison")
    
    # Extract unique values
    aggregation_methods = sorted(experiment_results['aggregation_name'].unique())
    attack_types = sorted(experiment_results['malicious_type'].unique())
    malicious_counts = sorted(experiment_results['malicious_count'].unique())
    bias_values = sorted(experiment_results['bias_values'].unique())
    
    # Create filters for the visualization in a container
    filter_container = st.container()
    
    with filter_container:
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter for aggregation methods (allow multiple selection)
            selected_aggregations = st.multiselect(
                "Select Aggregation Methods",
                options=aggregation_methods,
                default=aggregation_methods[:2] if len(aggregation_methods) > 1 else aggregation_methods,
                key="detailed_agg_selector"
            )
            
            # Filter for malicious counts (allow multiple selection)
            selected_mal_counts = st.multiselect(
                "Select Malicious Counts",
                options=malicious_counts,
                default=malicious_counts,
                key="detailed_mal_count_selector"
            )
        
        with col2:
            # Filter for attack types (allow multiple selection)
            selected_attacks = st.multiselect(
                "Select Attack Types",
                options=attack_types,
                default=attack_types[:2] if len(attack_types) > 1 else attack_types,
                key="detailed_attack_selector"
            )
            
            # Filter for bias values (allow multiple selection)
            selected_bias_values = st.multiselect(
                "Select Bias Values",
                options=bias_values,
                default=bias_values,
                key="detailed_bias_selector"
            )
    
    # Apply filters and get the latest round for each configuration
    if (selected_aggregations and selected_attacks and 
        selected_mal_counts and selected_bias_values):
        
        # Filter based on selections
        filtered_results = experiment_results[
            (experiment_results['aggregation_name'].isin(selected_aggregations)) &
            (experiment_results['malicious_type'].isin(selected_attacks)) &
            (experiment_results['malicious_count'].isin(selected_mal_counts)) &
            (experiment_results['bias_values'].isin(selected_bias_values))
        ]
        
        if filtered_results.empty:
            st.warning("No data available for the selected combination of filters.")
        else:
            # Get the latest round for each configuration
            # Group by all the dimensions we care about
            grouped = filtered_results.groupby([
                'aggregation_name', 'malicious_type', 'malicious_count', 'bias_values'
            ])
            
            latest_rounds = []
            for name, group in grouped:
                # Get the row with the maximum round_num
                latest_round = group.loc[group['round_num'].idxmax()]
                latest_rounds.append(latest_round)
            
            if not latest_rounds:
                st.warning("No data available after processing.")
            else:
                # Create a DataFrame with the results
                results_df = pd.DataFrame(latest_rounds)
                
                # Add a combined column for visualization labels
                results_df['config'] = results_df.apply(
                    lambda row: f"{row['aggregation_name']} | {row['malicious_type']} | {row['malicious_count']} | {row['bias_values']}",
                    axis=1
                )
                
                # Sort by aggregation method, attack type, etc.
                results_df = results_df.sort_values(
                    by=['aggregation_name', 'malicious_type', 'malicious_count', 'bias_values']
                )
                
                # Display a table with the raw data
                with st.expander("View Raw Data Table"):
                    display_columns = [
                        'aggregation_name', 'malicious_type', 'malicious_count', 
                        'bias_values', 'round_num', 'round_accuracy'
                    ]
                    
                    # Add backdoor success rate if available
                    if 'round_backdoor_success' in results_df.columns:
                        display_columns.append('round_backdoor_success')
                    
                    # Show the table
                    st.dataframe(
                        results_df[display_columns].sort_values(
                            by=['aggregation_name', 'malicious_type', 'malicious_count', 'bias_values']
                        ),
                        use_container_width=True
                    )
                
                # Create visualization tabs
                viz_tab1, viz_tab2 = st.tabs(["Model Accuracy", "Backdoor Success Rate"])
                
                with viz_tab1:
                    # Create a grouped bar chart for model accuracy
                    fig = px.bar(
                        results_df,
                        x='config',
                        y='round_accuracy',
                        color='aggregation_name',
                        barmode='group',
                        title='Model Accuracy by Configuration',
                        labels={
                            'config': 'Configuration', 
                            'round_accuracy': 'Model Accuracy', 
                            'aggregation_name': 'Aggregation Method'
                        },
                    )
                    
                    # Update layout for better readability
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        xaxis_title="",
                        yaxis_title="Model Accuracy",
                        legend_title="Aggregation Method",
                        height=600
                    )
                    
                    # Format y-axis as percentage
                    fig.update_yaxes(tickformat=".0%")
                    
                    # Show the plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a horizontal line chart for easier comparison
                    st.subheader("Horizontal View")
                    
                    fig_h = px.bar(
                        results_df,
                        y='config',
                        x='round_accuracy',
                        color='aggregation_name',
                        barmode='group',
                        orientation='h',
                        title='Model Accuracy by Configuration (Horizontal)',
                        labels={
                            'config': 'Configuration', 
                            'round_accuracy': 'Model Accuracy', 
                            'aggregation_name': 'Aggregation Method'
                        },
                    )
                    
                    # Update layout for better readability
                    fig_h.update_layout(
                        yaxis_title="",
                        xaxis_title="Model Accuracy",
                        legend_title="Aggregation Method",
                        height=max(400, len(results_df['config'].unique()) * 30)
                    )
                    
                    # Format x-axis as percentage
                    fig_h.update_xaxes(tickformat=".0%")
                    
                    # Show the plot
                    st.plotly_chart(fig_h, use_container_width=True)
                
                # Only show backdoor tab if data exists
                if 'round_backdoor_success' in results_df.columns:
                    with viz_tab2:
                        # Create a subset of data with backdoor success (not all attacks have it)
                        backdoor_df = results_df.dropna(subset=['round_backdoor_success'])
                        
                        if backdoor_df.empty:
                            st.info("No backdoor success data available for the selected configurations.")
                        else:
                            # Create a bar chart for backdoor success
                            fig = px.bar(
                                backdoor_df,
                                x='config',
                                y='round_backdoor_success',
                                color='aggregation_name',
                                barmode='group',
                                title='Backdoor Success Rate by Configuration',
                                labels={
                                    'config': 'Configuration', 
                                    'round_backdoor_success': 'Backdoor Success Rate', 
                                    'aggregation_name': 'Aggregation Method'
                                },
                            )
                            
                            # Update layout for better readability
                            fig.update_layout(
                                xaxis_tickangle=-45,
                                xaxis_title="",
                                yaxis_title="Backdoor Success Rate",
                                legend_title="Aggregation Method",
                                height=600
                            )
                            
                            # Format y-axis as percentage
                            fig.update_yaxes(tickformat=".0%")
                            
                            # Show the plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add a horizontal bar chart
                            st.subheader("Horizontal View")
                            
                            fig_h = px.bar(
                                backdoor_df,
                                y='config',
                                x='round_backdoor_success',
                                color='aggregation_name',
                                barmode='group',
                                orientation='h',
                                title='Backdoor Success Rate by Configuration (Horizontal)',
                                labels={
                                    'config': 'Configuration', 
                                    'round_backdoor_success': 'Backdoor Success Rate', 
                                    'aggregation_name': 'Aggregation Method'
                                },
                            )
                            
                            # Update layout for better readability
                            fig_h.update_layout(
                                yaxis_title="",
                                xaxis_title="Backdoor Success Rate",
                                legend_title="Aggregation Method",
                                height=max(400, len(backdoor_df['config'].unique()) * 30)
                            )
                            
                            # Format x-axis as percentage
                            fig_h.update_xaxes(tickformat=".0%")
                            
                            # Show the plot
                            st.plotly_chart(fig_h, use_container_width=True)
                else:
                    with viz_tab2:
                        st.info("No backdoor success data available in the dataset.")
    else:
        st.info("Please select at least one option for each filter to view the visualization.")
else:
    st.warning("No experiment results available to compare.")

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | 2. Detailed Performance Analysis")