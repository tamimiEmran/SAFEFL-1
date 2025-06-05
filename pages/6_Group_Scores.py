import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sidebar_template import create_sidebar

# Define render_analysis function first so it's available when needed
def render_analysis(group_scores, summary_stats, user_membership, experiment_results, attack_type=None, selected_exp_id=None, selected_nbyz=None, selected_bias=None):
    """
    Render the analysis for the given group scores data
    
    Parameters:
    - group_scores: DataFrame containing group score data
    - summary_stats: DataFrame containing summary statistics
    - user_membership: DataFrame containing user membership data
    - experiment_results: DataFrame containing experiment results
    - attack_type: Optional name of the attack type for display purposes
    - selected_exp_id: Selected experiment ID
    - selected_nbyz: Selected number of malicious users
    - selected_bias: Selected bias value
    """
    # Check if we need to filter the data by experiment parameters
    if experiment_results is not None and selected_exp_id is not None:
        # Get the experiment results data for this attack type and experiment ID
        exp_results = experiment_results[
            (experiment_results['malicious_type'] == attack_type) & 
            (experiment_results['experiment_id'] == selected_exp_id)
        ]
        
        # Further filter by nbyz if specified
        if selected_nbyz is not None:
            exp_results = exp_results[exp_results['malicious_count'] == selected_nbyz]
        
        # Further filter by bias if specified
        if selected_bias is not None:
            exp_results = exp_results[exp_results['bias_values'] == selected_bias]
        
        # Get the round numbers from the filtered experiment results
        if not exp_results.empty:
            matching_rounds = exp_results['round_num'].unique()
            
            # Filter group scores to only include these rounds
            group_scores = group_scores[group_scores['round'].isin(matching_rounds)]
    
    # Return if no data after filtering
    if group_scores.empty:
        st.warning(f"No group score data available for the selected filters.")
        return
    
    # Display description of current filter
    filter_description = f"Analysis for {attack_type}"
    if selected_nbyz is not None:
        filter_description += f" with {selected_nbyz} malicious users"
    if selected_bias is not None:
        filter_description += f" and bias value {selected_bias}"
    if selected_exp_id is not None:
        filter_description += f" (Experiment {selected_exp_id})"
    
    st.subheader(filter_description)
    
    # Overview section with key metrics
    st.subheader("Malicious Users Impact Overview")
    
    # Calculate high-level metrics
    total_groups = len(group_scores['group_id'].unique())
    total_rounds = len(group_scores['round'].unique())
    
    # Display high-level metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Groups", f"{total_groups}")
    
    with col2:
        st.metric("Total Rounds", f"{total_rounds}")
    
    with col3:
        max_malicious = group_scores['actual_malicious_count'].max()
        st.metric("Max Malicious Users in a Group", f"{max_malicious}")
    
    # Score by Malicious User Count Analysis
    st.subheader("Group Score by Malicious User Count")
    
    # Add round selection
    latest_round = group_scores['round'].max()
    selected_round = st.slider(
        "Select Round for Analysis", 
        min_value=int(group_scores['round'].min()),
        max_value=int(latest_round),
        value=int(latest_round),
        step=1,
        key=f"round_slider_{attack_type}_{selected_exp_id}_{selected_nbyz}_{selected_bias}"
    )
    
    # Filter data for the selected round
    round_scores = group_scores[group_scores['round'] == selected_round]
    
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs([
        "Current Round", 
        "All Rounds", 
        "Trend Analysis"
    ])
    
    with tab1:
        # Group by malicious count and calculate average score for selected round
        avg_scores = round_scores.groupby('actual_malicious_count')['score'].mean().reset_index()
        avg_scores = avg_scores.sort_values('actual_malicious_count')
        
        # Add count of groups in each category for reference
        group_counts = round_scores.groupby('actual_malicious_count').size().reset_index(name='count')
        avg_scores = pd.merge(avg_scores, group_counts, on='actual_malicious_count')
        
        # Create labels that include the count of groups
        avg_scores['label'] = avg_scores['actual_malicious_count'].astype(str) + ' mal users (' + avg_scores['count'].astype(str) + ' groups)'
        
        # Create the bar plot
        fig = px.bar(
            avg_scores,
            x='label',
            y='score',
            title=f'Average Group Score by Number of Malicious Users (Round {selected_round})',
            labels={'label': 'Number of Malicious Users', 'score': 'Average Group Score'},
            color='score',
            color_continuous_scale='RdBu_r'  # Red for low scores, blue for high scores
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Number of Malicious Users (Groups Count)",
            yaxis_title="Average Group Score",
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(title="Score")
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanatory text
        st.markdown("""
        This chart shows how the average group score is affected by the number of malicious 
        users present in the group for the selected round. Lower scores typically indicate 
        groups with more malicious activity detected by the system.
        """)
    
    with tab2:
        # Group by malicious count and calculate average score across all rounds
        avg_all_scores = group_scores.groupby('actual_malicious_count')['score'].mean().reset_index()
        avg_all_scores = avg_all_scores.sort_values('actual_malicious_count')
        
        # Add count of groups in each category for reference
        group_all_counts = group_scores.groupby('actual_malicious_count').size().reset_index(name='count')
        avg_all_scores = pd.merge(avg_all_scores, group_all_counts, on='actual_malicious_count')
        
        # Create labels that include the count of groups
        avg_all_scores['label'] = avg_all_scores['actual_malicious_count'].astype(str) + ' mal users (' + avg_all_scores['count'].astype(str) + ' groups)'
        
        # Create the bar plot
        fig = px.bar(
            avg_all_scores,
            x='label',
            y='score',
            title=f'Average Group Score by Number of Malicious Users (All Rounds)',
            labels={'label': 'Number of Malicious Users', 'score': 'Average Group Score'},
            color='score',
            color_continuous_scale='RdBu_r'
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_title="Number of Malicious Users (Groups Count)",
            yaxis_title="Average Group Score",
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(title="Score")
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add round range filter
        if st.checkbox("Filter by round range", key=f"checkbox_round_range_{attack_type}_{selected_exp_id}_{selected_nbyz}_{selected_bias}"):
            min_round = int(group_scores['round'].min())
            max_round = int(group_scores['round'].max())
            
            round_range = st.slider(
                "Select Round Range for Analysis",
                min_value=min_round,
                max_value=max_round,
                value=(min_round, max_round),
                step=1,
                key=f"all_rounds_slider_{attack_type}_{selected_exp_id}_{selected_nbyz}_{selected_bias}"
            )
            
            # Filter data by selected round range
            filtered_data = group_scores[
                (group_scores['round'] >= round_range[0]) & 
                (group_scores['round'] <= round_range[1])
            ]
            
            # Group by malicious count and calculate average score for filtered rounds
            avg_filtered_scores = filtered_data.groupby('actual_malicious_count')['score'].mean().reset_index()
            avg_filtered_scores = avg_filtered_scores.sort_values('actual_malicious_count')
            
            # Add count of groups in each category for reference
            group_filtered_counts = filtered_data.groupby('actual_malicious_count').size().reset_index(name='count')
            avg_filtered_scores = pd.merge(avg_filtered_scores, group_filtered_counts, on='actual_malicious_count')
            
            # Create labels that include the count of groups
            avg_filtered_scores['label'] = avg_filtered_scores['actual_malicious_count'].astype(str) + ' mal users (' + avg_filtered_scores['count'].astype(str) + ' groups)'
            
            # Create the bar plot
            fig = px.bar(
                avg_filtered_scores,
                x='label',
                y='score',
                title=f'Average Group Score by Number of Malicious Users (Rounds {round_range[0]} to {round_range[1]})',
                labels={'label': 'Number of Malicious Users', 'score': 'Average Group Score'},
                color='score',
                color_continuous_scale='RdBu_r'
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title="Number of Malicious Users (Groups Count)",
                yaxis_title="Average Group Score",
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(title="Score")
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Trend analysis of scores by malicious count over rounds
        st.subheader("Score Trend by Malicious User Count")
        
        # Get the unique counts of malicious users
        malicious_counts = sorted(group_scores['actual_malicious_count'].unique())
        
        # Let user select specific malicious counts to visualize
        selected_counts = st.multiselect(
            "Select number of malicious users to visualize trends",
            options=malicious_counts,
            default=malicious_counts[:3] if len(malicious_counts) > 2 else malicious_counts,
            key=f"malicious_counts_{attack_type}_{selected_exp_id}_{selected_nbyz}_{selected_bias}"
        )
        
        if selected_counts:
            # Calculate average score per round for each selected malicious count
            trend_data = []
            for count in selected_counts:
                subset = group_scores[group_scores['actual_malicious_count'] == count]
                round_avgs = subset.groupby('round')['score'].mean().reset_index()
                round_avgs['malicious_count'] = count
                trend_data.append(round_avgs)
            
            if trend_data:
                trend_df = pd.concat(trend_data)
                
                # Create line chart
                fig = px.line(
                    trend_df,
                    x='round',
                    y='score',
                    color='malicious_count',
                    title=f'Group Score Trend by Number of Malicious Users',
                    labels={
                        'round': 'Round', 
                        'score': 'Average Group Score',
                        'malicious_count': 'Malicious Users'
                    },
                    line_shape='spline'
                )
                
                # Customize layout
                fig.update_layout(
                    xaxis_title="Round",
                    yaxis_title="Average Group Score",
                    hovermode="x unified",
                    legend_title="Malicious Users"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                This chart shows how the average score for groups with different numbers of malicious users
                changes over training rounds. This helps identify if the federated learning system becomes
                better at identifying and penalizing groups with malicious users over time.
                """)
            else:
                st.warning("No data available for the selected malicious counts.")
        else:
            st.info("Please select at least one malicious count to visualize trends.")
    
    # Malicious Filter Effectiveness
    st.subheader("Malicious User Filtering by Group Score")
    
    # Calculate filtering thresholds and effectiveness
    if user_membership is not None and not user_membership.empty:
        # Filter user_membership to match the attack type if applicable
        if attack_type and 'attack_type' in user_membership.columns:
            filtered_membership = user_membership[user_membership['attack_type'] == attack_type]
            
            # Further filter by experiment ID if specified
            if selected_exp_id is not None:
                filtered_membership = filtered_membership[filtered_membership['experiment_id'] == selected_exp_id]
                
            # If we're filtering by nbyz and bias, we need to use the round information from experiment results
            if selected_nbyz is not None or selected_bias is not None:
                if experiment_results is not None:
                    exp_filter = experiment_results['malicious_type'] == attack_type
                    exp_filter &= experiment_results['experiment_id'] == selected_exp_id
                    
                    if selected_nbyz is not None:
                        exp_filter &= experiment_results['malicious_count'] == selected_nbyz
                        
                    if selected_bias is not None:
                        exp_filter &= experiment_results['bias_values'] == selected_bias
                        
                    matching_rounds = experiment_results[exp_filter]['round_num'].unique()
                    filtered_membership = filtered_membership[filtered_membership['round'].isin(matching_rounds)]
        else:
            filtered_membership = user_membership
        
        # Filter for selected round
        round_membership = filtered_membership[filtered_membership['round'] == selected_round]
        
        # Join with group scores
        user_group_data = pd.merge(
            round_membership,
            round_scores[['group_id', 'score']],
            on='group_id',
            how='inner'  # Use inner join to ensure we only keep matching records
        )
        
        if not user_group_data.empty:
            # Create scatter plot of group scores vs filtering effectiveness
            fig = px.scatter(
                user_group_data.groupby('group_id').agg({
                    'score': 'first',
                    'is_actually_malicious': 'sum',
                    'is_filtered_out': 'sum',
                    'user_id': 'count'
                }).reset_index(),
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
                coloraxis_colorbar=dict(title="Malicious Users")
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation statistics
            try:
                correlation = user_group_data.groupby('group_id').agg({
                    'score': 'first',
                    'is_actually_malicious': 'sum',
                    'is_filtered_out': 'sum'
                }).corr().loc['score', 'is_filtered_out']
                
                st.metric("Correlation between Group Score and Filtered Users", f"{correlation:.2f}")
            except:
                st.warning("Unable to calculate correlation due to insufficient data points.")
            
            st.markdown("""
            This scatter plot shows the relationship between group scores and the number of users filtered out.
            Each point represents a group, with the size indicating the number of users in the group and the
            color showing how many malicious users are actually in that group.
            
            A strong negative correlation indicates that the system is effectively identifying and filtering
            malicious users, as lower group scores should correspond to more filtered users.
            """)
        else:
            st.warning("No matching user membership data for the selected round and/or attack type.")
    else:
        st.warning("No user membership data available for filtering effectiveness analysis.")
    
    # Group Score Distribution Analysis
    st.subheader("Group Score Distribution by Malicious User Count")
    
    # Create box plot of scores by malicious count
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
        showlegend=False
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanatory text
    st.markdown("""
    This box plot shows the distribution of group scores for each number of malicious users.
    This visualization helps understand not just the average impact of malicious users on scores,
    but also the variability and potential outliers.
    
    A good detection system should show clear separation between boxes, indicating that groups
    with more malicious users consistently receive lower scores.
    """)
    
    # Raw Data Exploration
    expander_label = "Explore Raw Group Score Data"
    if attack_type:
        expander_label += f" ({attack_type})"
    with st.expander(expander_label):
        # Add a search/filter option
        search_group = st.number_input(
            "Filter by Group ID", 
            min_value=0, 
            max_value=int(round_scores['group_id'].max()) if not round_scores.empty else 0, 
            step=1,
            key=f"group_filter_{attack_type}_{selected_exp_id}_{selected_nbyz}_{selected_bias}"
        )
        
        if search_group is not None and not round_scores.empty:
            filtered_group_data = round_scores[round_scores['group_id'] == search_group]
            st.dataframe(filtered_group_data)
        else:
            st.dataframe(round_scores)

# Set page configuration
st.set_page_config(
    page_title="Group Scores - Hierarchical Federated Learning",
    page_icon="📊",
    layout="wide"
)

# Ensure data loader is available
if 'data_loader' not in st.session_state:
    st.error("Please navigate to the main page first to initialize the data loader.")
    st.stop()

# Page title
st.title("Group Scores Analysis")
st.markdown("Analyze the relationship between malicious users and group scores")

# Display current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.info(f"Currently viewing Experiment ID: {current_exp_id}")

# Create consistent sidebar
create_sidebar()

# Get data
group_scores = st.session_state.data_loader.get_group_scores()
summary_stats = st.session_state.data_loader.get_summary_stats()
user_membership = st.session_state.data_loader.get_user_membership()
experiment_results = st.session_state.data_loader.get_experiment_results()

if group_scores is not None and not group_scores.empty:
    # Check if attack_type column exists
    has_attack_type = 'attack_type' in group_scores.columns
    
    if has_attack_type:
        # Get unique attack types and create a tab for each
        attack_types = sorted(group_scores['attack_type'].unique())
        
        # Create tabs for each attack type
        attack_tabs = st.tabs(attack_types)
        
        for i, attack_type in enumerate(attack_types):
            with attack_tabs[i]:
                # Filter by attack type
                filtered_scores = group_scores[group_scores['attack_type'] == attack_type]
                
                # Create a sidebar for filtering options
                st.sidebar.markdown(f"## Filter Options for {attack_type}")
                
                # Add experiment ID filter
                attack_exp_ids = sorted(filtered_scores['experiment_id'].unique())
                selected_exp_id = st.sidebar.selectbox(
                    "Select Experiment ID:",
                    options=attack_exp_ids,
                    index=0,
                    key=f"exp_id_select_{attack_type}"
                )
                
                # Get nbyz and bias values for the selected experiment and attack
                nbyz_values = []
                bias_values = []
                
                if experiment_results is not None:
                    exp_attack_results = experiment_results[
                        (experiment_results['malicious_type'] == attack_type) & 
                        (experiment_results['experiment_id'] == selected_exp_id)
                    ]
                    nbyz_values = sorted(exp_attack_results['malicious_count'].unique())
                    bias_values = sorted(exp_attack_results['bias_values'].unique())
                
                # Add nbyz filter if there are multiple values
                selected_nbyz = None
                if len(nbyz_values) > 1:
                    selected_nbyz = st.sidebar.selectbox(
                        "Select Number of Malicious Users:",
                        options=nbyz_values,
                        index=0,
                        key=f"nbyz_select_{attack_type}"
                    )
                
                # Add bias filter if there are multiple values
                selected_bias = None
                if len(bias_values) > 1:
                    selected_bias = st.sidebar.selectbox(
                        "Select Bias Value:",
                        options=bias_values,
                        index=0,
                        key=f"bias_select_{attack_type}"
                    )
                
                # Render analysis for this attack type with filters
                render_analysis(
                    filtered_scores, 
                    summary_stats, 
                    user_membership, 
                    experiment_results,
                    attack_type=attack_type,
                    selected_exp_id=selected_exp_id,
                    selected_nbyz=selected_nbyz,
                    selected_bias=selected_bias
                )
    else:
        # No attack_type column, just render the analysis with all data
        render_analysis(group_scores, summary_stats, user_membership, experiment_results)
else:
    st.warning("No group score data available to visualize.")

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | Group Scores Analysis")