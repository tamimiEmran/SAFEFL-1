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
    page_title="5. Detailed User Scores - Hierarchical Federated Learning",
    page_icon="👥",
    layout="wide"
)

# Ensure data loader is available
if 'data_loader' not in st.session_state:
    st.error("Please navigate to the main page first to initialize the data loader.")
    st.stop()

# Page title
st.title("5. Detailed User Scores Analysis")
st.markdown("Analyze user scores by combination of attack type, malicious count, and bias value")

# Display current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.info(f"Currently viewing Experiment ID: {current_exp_id}")

# Create consistent sidebar
create_sidebar()

# Get data
user_scores = st.session_state.data_loader.get_user_scores()
user_membership = st.session_state.data_loader.get_user_membership()
experiment_results = st.session_state.data_loader.get_experiment_results()

# Create the detailed visualization section
st.header("User Score Analysis by Configuration")

if user_scores is not None and not user_scores.empty and experiment_results is not None and not experiment_results.empty:
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
                key="detailed_attack_selector"
            )
        
        with col2:
            # Filter for malicious counts
            selected_mal_count = st.selectbox(
                "Select Malicious Count",
                options=malicious_counts,
                key="detailed_mal_count_selector"
            )
        
        with col3:
            # Filter for bias values
            selected_bias = st.selectbox(
                "Select Bias Value",
                options=bias_values,
                key="detailed_bias_selector"
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
        
        # If we have user scores with the attack_type field
        if 'attack_type' in user_scores.columns:
            # Filter user scores to match the attack type and rounds
            filtered_scores = user_scores[
                (user_scores['attack_type'] == selected_attack) &
                (user_scores['round'].isin(matching_rounds))
            ]
        else:
            # Just filter by rounds if attack_type isn't available
            filtered_scores = user_scores[user_scores['round'].isin(matching_rounds)]
        
        if filtered_scores.empty:
            st.warning(f"No user score data available for the selected configuration")
        else:
            # Add round selector
            available_rounds = sorted(filtered_scores['round'].unique())
            selected_round = st.select_slider(
                "Select Round",
                options=available_rounds,
                value=available_rounds[-1] if available_rounds else None,
                key="round_selector"
            )
            
            # Filter to the selected round
            round_scores = filtered_scores[filtered_scores['round'] == selected_round]
            
            # Configuration information
            st.subheader(f"Configuration: {selected_attack}, Malicious Count: {selected_mal_count}, Bias: {selected_bias}, Round: {selected_round}")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                malicious_users = round_scores[round_scores['is_actually_malicious'] == 1]
                benign_users = round_scores[round_scores['is_actually_malicious'] == 0]
                
                st.metric(
                    "Total Users", 
                    f"{len(round_scores)}", 
                    help="Total number of users in this round"
                )
            
            with col2:
                st.metric(
                    "Malicious Users", 
                    f"{len(malicious_users)}", 
                    help="Number of users that are actually malicious"
                )
            
            with col3:
                filtered_users = round_scores[round_scores['is_filtered_out'] == 1]
                filtered_malicious = filtered_users[filtered_users['is_actually_malicious'] == 1]
                
                recall = len(filtered_malicious) / len(malicious_users) if len(malicious_users) > 0 else 0
                
                st.metric(
                    "Filtered Users", 
                    f"{len(filtered_users)}", 
                    delta=f"{len(filtered_malicious)} malicious",
                    help="Number of users filtered out by the system"
                )
            
            with col4:
                precision = len(filtered_malicious) / len(filtered_users) if len(filtered_users) > 0 else 0
                
                st.metric(
                    "Detection Recall", 
                    f"{recall:.2%}", 
                    help="Percentage of malicious users that were detected"
                )
            
            # Create a visualization of user scores
            st.subheader("User Score Distribution")
            
            # Add user type filter
            user_type = st.radio(
                "Filter by User Type",
                options=["All Users", "Malicious Users", "Benign Users"],
                horizontal=True,
                key="user_type_selector"
            )
            
            # Apply user type filter
            plot_data = round_scores
            if user_type == "Malicious Users":
                plot_data = malicious_users
            elif user_type == "Benign Users":
                plot_data = benign_users
            
            # Create histogram of user scores
            fig = px.histogram(
                plot_data,
                x="score",
                color="is_actually_malicious",
                marginal="box",
                hover_data=["user_id", "is_filtered_out"],
                title=f"Distribution of User Scores for {user_type}",
                labels={"score": "User Score", "is_actually_malicious": "Is Malicious"},
                color_discrete_map={0: "green", 1: "red"},
                nbins=30
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="User Score",
                yaxis_title="Count",
                legend_title="User Type",
                height=500
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot of scores vs filtered status
            st.subheader("User Scores vs Filtering Status")
            
            fig = px.scatter(
                plot_data,
                x="score",
                y="is_filtered_out",
                color="is_actually_malicious",
                hover_data=["user_id"],
                title=f"User Scores vs Filtering Status for {user_type}",
                labels={
                    "score": "User Score", 
                    "is_filtered_out": "Filtered Out", 
                    "is_actually_malicious": "Is Malicious"
                },
                color_discrete_map={0: "green", 1: "red"},
            )
            
            # Add horizontal jitter to better visualize overlapping points
            fig.update_traces(marker=dict(size=10), jitter=0.3)
            
            # Update layout
            fig.update_layout(
                xaxis_title="User Score",
                yaxis_title="Filtered Out",
                legend_title="User Type",
                height=500,
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1],
                    ticktext=['No', 'Yes']
                )
            )
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display raw data table in an expander
            with st.expander("View Raw User Score Data"):
                # Add a search/filter option
                search_user = st.number_input(
                    "Filter by User ID", 
                    min_value=0, 
                    max_value=int(round_scores['user_id'].max()) if not round_scores.empty else 0, 
                    step=1,
                    key="user_filter"
                )
                
                if search_user is not None and not round_scores.empty:
                    filtered_user_data = round_scores[round_scores['user_id'] == search_user]
                    st.dataframe(filtered_user_data)
                else:
                    # Show the first 100 users to avoid performance issues
                    display_data = round_scores.head(100) if len(round_scores) > 100 else round_scores
                    st.dataframe(display_data)
                    
                    if len(round_scores) > 100:
                        st.info(f"Showing first 100 of {len(round_scores)} users. Use the filter above to see specific users.")
else:
    st.warning("No data available for analysis. Please ensure that experiment results and user scores are available.")

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | 5. Detailed User Scores Analysis")