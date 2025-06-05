import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sidebar_template import create_sidebar

# Set page configuration
st.set_page_config(
    page_title="7. Detailed Group Scores - Hierarchical Federated Learning",
    page_icon="🔍",
    layout="wide"
)

# Ensure data loader is available
if 'data_loader' not in st.session_state:
    st.error("Please navigate to the main page first to initialize the data loader.")
    st.stop()

# Page title
st.title("7. Detailed Group Score Analysis")
st.markdown("Analyze group scores by combination of attack type, malicious count, and bias value")

# Display current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.info(f"Currently viewing Experiment ID: {current_exp_id}")

# Create consistent sidebar
create_sidebar()

# Get data
group_scores = st.session_state.data_loader.get_group_scores()
user_membership = st.session_state.data_loader.get_user_membership()
experiment_results = st.session_state.data_loader.get_experiment_results()

# Create the detailed visualization section
st.header("Group Score Analysis by Configuration")

if group_scores is not None and not group_scores.empty and experiment_results is not None and not experiment_results.empty:
    # Extract unique values from experiment results
    attack_types = sorted(experiment_results['malicious_type'].unique())
    malicious_counts = sorted(experiment_results['malicious_count'].unique())
    bias_values = sorted(experiment_results['bias_values'].unique())
    
    # Create filters for the visualization
    filter_container = st.container()
    
    with filter_container:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter for attack types
            selected_attack = st.selectbox(
                "Select Attack Type",
                options=attack_types,
                key="detailed_group_attack_selector"
            )
        
        with col2:
            # Filter for malicious counts
            selected_mal_count = st.selectbox(
                "Select Malicious Count",
                options=malicious_counts,
                key="detailed_group_mal_count_selector"
            )
        
        with col3:
            # Filter for bias values
            selected_bias = st.selectbox(
                "Select Bias Value",
                options=bias_values,
                key="detailed_group_bias_selector"
            )
    
    # Filter experiment results to get rounds that match the selected configuration
    filtered_exp = experiment_results[
        (experiment_results['malicious_type'] == selected_attack) &
        (experiment_results['malicious_count'] == selected_mal_count) &
        (experiment_results['bias_values'] == selected_bias)
    ]
    
    if filtered_exp.empty:
        st.warning(f"No experiment data available for the selected configuration: {selected_attack}, malicious count: {selected_mal_count}, bias: {selected_bias}")
    else:
        # Get the rounds from the filtered experiment results
        matching_rounds = filtered_exp['round_num'].unique()
        
        # If we have group scores with the attack_type field
        if 'attack_type' in group_scores.columns:
            # Filter group scores to match the attack type and rounds
            filtered_scores = group_scores[
                (group_scores['attack_type'] == selected_attack) &
                (group_scores['round'].isin(matching_rounds))
            ]
        else:
            # Just filter by rounds if attack_type isn't available
            filtered_scores = group_scores[group_scores['round'].isin(matching_rounds)]
        
        if filtered_scores.empty:
            st.warning(f"No group score data available for the selected configuration")
        else:
            # Add round selector
            available_rounds = sorted(filtered_scores['round'].unique())
            selected_round = st.select_slider(
                "Select Round",
                options=available_rounds,
                value=available_rounds[-1] if available_rounds else None,
                key="group_round_selector"
            )
            
            # Filter to the selected round
            round_scores = filtered_scores[filtered_scores['round'] == selected_round]
            
            # Configuration information
            st.subheader(f"Configuration: {selected_attack}, Malicious Count: {selected_mal_count}, Bias: {selected_bias}, Round: {selected_round}")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Groups", 
                    f"{len(round_scores)}", 
                    help="Total number of groups in this round"
                )
            
            with col2:
                groups_with_malicious = round_scores[round_scores['actual_malicious_count'] > 0]
                st.metric(
                    "Groups With Malicious Users", 
                    f"{len(groups_with_malicious)}", 
                    help="Number of groups containing at least one malicious user"
                )
            
            with col3:
                avg_score = round_scores['score'].mean()
                st.metric(
                    "Average Group Score", 
                    f"{avg_score:.2f}", 
                    help="Average score across all groups"
                )
            
            with col4:
                max_malicious = round_scores['actual_malicious_count'].max()
                st.metric(
                    "Max Malicious Users in a Group", 
                    f"{max_malicious}", 
                    help="Maximum number of malicious users in any group"
                )
            
            # Create a bar chart of group scores by malicious count
            st.subheader("Group Scores by Number of Malicious Users")
            
            # Group by malicious count and calculate mean score
            malicious_counts = sorted(round_scores['actual_malicious_count'].unique())
            avg_by_malicious = round_scores.groupby('actual_malicious_count')['score'].mean().reset_index()
            count_by_malicious = round_scores.groupby('actual_malicious_count').size().reset_index(name='count')
            avg_by_malicious = pd.merge(avg_by_malicious, count_by_malicious, on='actual_malicious_count')
            
            # Create labels with count of groups
            avg_by_malicious['label'] = avg_by_malicious['actual_malicious_count'].astype(str) + ' mal users (' + avg_by_malicious['count'].astype(str) + ' groups)'
            
            # Create bar chart
            fig = px.bar(
                avg_by_malicious,
                x='label',
                y='score',
                title=f'Average Group Score by Number of Malicious Users (Round {selected_round})',
                labels={
                    'label': 'Number of Malicious Users (Group Count)',
                    'score': 'Average Group Score'
                },
                color='score',
                color_continuous_scale='RdBu_r'  # Red for low scores, blue for high scores
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Number of Malicious Users (Groups Count)",
                yaxis_title="Average Group Score",
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(title="Score"),
                height=500
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Create a box plot to show distribution of scores by malicious count
            st.subheader("Distribution of Group Scores by Number of Malicious Users")
            
            # Use raw data for box plot
            fig = px.box(
                round_scores,
                x='actual_malicious_count',
                y='score',
                title=f'Distribution of Group Scores by Malicious User Count (Round {selected_round})',
                labels={
                    'actual_malicious_count': 'Number of Malicious Users',
                    'score': 'Group Score'
                },
                color='actual_malicious_count',
                notched=True
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Number of Malicious Users",
                yaxis_title="Group Score",
                showlegend=False,
                height=500
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # If user membership data is available, create a scatter plot of groups
            if user_membership is not None and not user_membership.empty:
                st.subheader("Group Composition and Filtering Analysis")
                
                # Filter user membership to match the selected round and attack type
                if 'attack_type' in user_membership.columns:
                    filtered_membership = user_membership[
                        (user_membership['attack_type'] == selected_attack) &
                        (user_membership['round'] == selected_round)
                    ]
                else:
                    filtered_membership = user_membership[user_membership['round'] == selected_round]
                
                if not filtered_membership.empty:
                    # Join with group scores
                    user_group_data = pd.merge(
                        filtered_membership,
                        round_scores[['group_id', 'score']],
                        on='group_id',
                        how='inner'
                    )
                    
                    if not user_group_data.empty:
                        # Group by group_id and calculate metrics
                        group_metrics = user_group_data.groupby('group_id').agg({
                            'score': 'first',
                            'is_actually_malicious': 'sum',
                            'is_filtered_out': 'sum',
                            'user_id': 'count'
                        }).reset_index()
                        
                        # Create a scatter plot
                        fig = px.scatter(
                            group_metrics,
                            x='score',
                            y='is_filtered_out',
                            size='user_id',
                            color='is_actually_malicious',
                            hover_name='group_id',
                            title=f'Group Score vs. Filtered Users (Round {selected_round})',
                            labels={
                                'score': 'Group Score', 
                                'is_filtered_out': 'Number of Filtered Users',
                                'user_id': 'Group Size',
                                'is_actually_malicious': 'Number of Malicious Users'
                            }
                        )
                        
                        # Customize layout
                        fig.update_layout(
                            xaxis_title="Group Score",
                            yaxis_title="Number of Filtered Users",
                            coloraxis_colorbar=dict(title="Malicious Users"),
                            height=600
                        )
                        
                        # Show the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add correlation metric
                        try:
                            correlation = group_metrics['score'].corr(group_metrics['is_filtered_out'])
                            st.metric(
                                "Correlation between Group Score and Filtered Users", 
                                f"{correlation:.2f}",
                                help="Negative correlation indicates lower scores correlate with more filtered users"
                            )
                        except:
                            st.warning("Unable to calculate correlation due to insufficient data.")
                    else:
                        st.warning("No matching user membership data for the selected round.")
                else:
                    st.warning("No user membership data available for the selected round and/or attack type.")
            
            # Display raw data table in an expander
            with st.expander("View Raw Group Score Data"):
                # Add a search/filter option
                search_group = st.number_input(
                    "Filter by Group ID", 
                    min_value=0, 
                    max_value=int(round_scores['group_id'].max()) if not round_scores.empty else 0, 
                    step=1,
                    key="group_id_filter"
                )
                
                if search_group is not None and not round_scores.empty:
                    filtered_group_data = round_scores[round_scores['group_id'] == search_group]
                    st.dataframe(filtered_group_data)
                else:
                    st.dataframe(round_scores)
else:
    st.warning("No data available for analysis. Please ensure that experiment results and group scores are available.")

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | 7. Detailed Group Score Analysis")