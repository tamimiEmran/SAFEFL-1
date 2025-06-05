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

# Set page configuration
st.set_page_config(
    page_title="User Scores - Hierarchical Federated Learning",
    page_icon="👤",
    layout="wide"
)

# Ensure data loader is available
if 'data_loader' not in st.session_state:
    st.error("Please navigate to the main page first to initialize the data loader.")
    st.stop()

# Page title
st.title("User Scores Analysis")
st.markdown("Explore individual user scores and filtering effectiveness")

# Display current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.info(f"Currently viewing Experiment ID: {current_exp_id}")

# Create consistent sidebar
create_sidebar()

# Get data
user_scores = st.session_state.data_loader.get_user_scores()
user_membership = st.session_state.data_loader.get_user_membership()
summary_stats = st.session_state.data_loader.get_summary_stats()
experiment_results = st.session_state.data_loader.get_experiment_results()

if user_scores is not None and not user_scores.empty:
    # Get unique attack types
    attack_types = sorted(user_scores['attack_type'].unique()) if 'attack_type' in user_scores.columns else ['default']
    
    # Get unique experiment IDs
    experiment_ids = sorted(user_scores['experiment_id'].unique()) if 'experiment_id' in user_scores.columns else [None]
    
    # Create tabs for each attack type
    attack_tabs = st.tabs(attack_types)
    
    for i, attack_type in enumerate(attack_types):
        with attack_tabs[i]:
            # Filter by attack type
            filtered_scores = user_scores[user_scores['attack_type'] == attack_type]
            
            # Filter experiment results to get nbyz and bias values for this attack
            attack_results = experiment_results[experiment_results['malicious_type'] == attack_type] if experiment_results is not None else None
            
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
            
            # Filter by experiment ID
            filtered_scores = filtered_scores[filtered_scores['experiment_id'] == selected_exp_id]
            
            # Get nbyz and bias values for the selected experiment and attack
            nbyz_values = []
            bias_values = []
            
            if attack_results is not None:
                exp_attack_results = attack_results[attack_results['experiment_id'] == selected_exp_id]
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
                
                # Join with experiment results to get the needed columns for filtering
                if attack_results is not None:
                    # Create mapping from experiment ID and round to nbyz and bias
                    exp_rounds = exp_attack_results[['round_num', 'malicious_count', 'bias_values']].copy()
                    exp_rounds = exp_rounds.rename(columns={'round_num': 'round'})
                    
                    # Only keep rounds that match selected nbyz
                    matching_rounds = exp_rounds[exp_rounds['malicious_count'] == selected_nbyz]['round'].unique()
                    filtered_scores = filtered_scores[filtered_scores['round'].isin(matching_rounds)]
            
            # Add bias filter if there are multiple values
            selected_bias = None
            if len(bias_values) > 1:
                selected_bias = st.sidebar.selectbox(
                    "Select Bias Value:",
                    options=bias_values,
                    index=0,
                    key=f"bias_select_{attack_type}"
                )
                
                # Join with experiment results to get the needed columns for filtering
                if attack_results is not None:
                    # Create mapping from experiment ID and round to nbyz and bias
                    exp_rounds = exp_attack_results[['round_num', 'malicious_count', 'bias_values']].copy()
                    exp_rounds = exp_rounds.rename(columns={'round_num': 'round'})
                    
                    # Further filter rounds that match both nbyz and bias
                    if selected_nbyz is not None:
                        matching_rounds = exp_rounds[(exp_rounds['malicious_count'] == selected_nbyz) & 
                                                    (exp_rounds['bias_values'] == selected_bias)]['round'].unique()
                    else:
                        matching_rounds = exp_rounds[exp_rounds['bias_values'] == selected_bias]['round'].unique()
                    
                    filtered_scores = filtered_scores[filtered_scores['round'].isin(matching_rounds)]
            
            # Display description of current filter
            filter_description = f"Analysis for {attack_type}"
            if selected_nbyz is not None:
                filter_description += f" with {selected_nbyz} malicious users"
            if selected_bias is not None:
                filter_description += f" and bias value {selected_bias}"
            
            st.subheader(filter_description)
            
            if filtered_scores.empty:
                st.warning(f"No data available for {attack_type} with the selected filters.")
                continue
            
            # Allow user to select round for analysis
            latest_round = filtered_scores['round'].max()
            selected_round = st.slider(
                "Select Round for Analysis", 
                min_value=int(filtered_scores['round'].min()),
                max_value=int(latest_round),
                value=int(latest_round),
                step=1,
                key=f"round_slider_{attack_type}_{selected_exp_id}_{selected_nbyz}_{selected_bias}"
            )
            
            # Filter data for the selected round
            round_scores = filtered_scores[filtered_scores['round'] == selected_round]
            
            # User score overview statistics
            st.subheader(f"User Score Statistics for Round {selected_round}")
                
            # Display user score overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Average score for honest (benign) users
                benign_scores = round_scores[round_scores['is_actually_malicious'] == 0]['score']
                avg_benign_score = benign_scores.mean() if not benign_scores.empty else 0
                st.metric("Avg. Honest User Score", f"{avg_benign_score:.2f}")

            with col2:
                # Average score for malicious users
                malicious_scores = round_scores[round_scores['is_actually_malicious'] == 1]['score']
                avg_malicious_score = malicious_scores.mean() if not malicious_scores.empty else 0
                st.metric("Avg. Malicious User Score", f"{avg_malicious_score:.2f}")

            with col3:
                # Calculate correctly filtered users (malicious users that were filtered out)
                correctly_filtered = round_scores[(round_scores['is_actually_malicious'] == 1) & 
                                              (round_scores['is_filtered_out'] == 1)].shape[0]
                total_malicious = round_scores[round_scores['is_actually_malicious'] == 1].shape[0]
                
                # Display as fraction and percentage
                if total_malicious > 0:
                    correct_filter_ratio = correctly_filtered / total_malicious
                    st.metric("Correctly Filtered", f"{correctly_filtered}/{total_malicious} ({correct_filter_ratio:.0%})")
                else:
                    st.metric("Correctly Filtered", "0/0 (0%)")

            with col4:
                # Total count of malicious users
                st.metric("Total Malicious Users", f"{total_malicious}")
            
            # Create two main analysis tabs
            analysis_tab1, analysis_tab2 = st.tabs(["Recall Over Time", "Score Gap Analysis"])
            
            with analysis_tab1:
                # Recall over time visualization
                st.subheader("Recall Over Time")
                
                # Calculate recall for each round
                recall_data = filtered_scores.groupby('round').apply(
                    lambda x: (
                        x[(x['is_actually_malicious'] == 1) & (x['is_filtered_out'] == 1)].shape[0] / 
                        x[x['is_actually_malicious'] == 1].shape[0]
                    ) if x[x['is_actually_malicious'] == 1].shape[0] > 0 else 0
                ).reset_index(name='recall')
                
                # Create line chart for recall over time
                fig = px.line(
                    recall_data,
                    x='round',
                    y='recall',
                    title=f'Recall Over Training Rounds ({attack_type})',
                    labels={'round': 'Round', 'recall': 'Recall'},
                    line_shape='spline'
                )
                
                # Customize the line
                fig.update_traces(line=dict(width=2, color='green'))
                
                # Customize layout
                fig.update_layout(
                    xaxis_title="Round",
                    yaxis_title="Recall",
                    hovermode="x unified",
                    yaxis=dict(range=[0, 1])
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
            
            with analysis_tab2:
                # Score gap analysis (difference between benign and malicious scores)
                st.subheader("Benign-Malicious Score Gap")
                
                # Calculate the score gap per round
                score_gap = filtered_scores.groupby('round').apply(
                    lambda x: x[x['is_actually_malicious'] == 0]['score'].mean() - 
                             x[x['is_actually_malicious'] == 1]['score'].mean() if 
                    (not x[x['is_actually_malicious'] == 0].empty and not x[x['is_actually_malicious'] == 1].empty) else np.nan
                ).reset_index(name='score_gap')
                
                # Create line chart for score gap
                fig = px.line(
                    score_gap,
                    x='round',
                    y='score_gap',
                    title=f'Gap Between Average Benign and Malicious Scores ({attack_type})',
                    labels={
                        'round': 'Round', 
                        'score_gap': 'Score Gap (Benign - Malicious)'
                    },
                    line_shape='spline'
                )
                
                # Add a horizontal line at y=0 for reference
                fig.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray", 
                    annotation_text="No Differentiation"
                )
                
                # Customize the line
                fig.update_traces(line=dict(width=2, color='purple'))
                
                # Customize layout
                fig.update_layout(
                    xaxis_title="Round",
                    yaxis_title="Score Gap",
                    hovermode="x unified"
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
            
            # Add a section comparing different nbyz and bias values if there are multiple
            if len(nbyz_values) > 1 or len(bias_values) > 1:
                st.subheader("Comparative Analysis")
                
                # Create tabs for comparison
                comp_tab1, comp_tab2 = st.tabs(["By # of Malicious Users", "By Bias Value"])
                
                with comp_tab1:
                    if len(nbyz_values) > 1:
                        st.write("### Comparison by Number of Malicious Users")
                        
                        # Create a comparison dataframe for the selected round across nbyz values
                        nbyz_comparison = []
                        
                        for nbyz in nbyz_values:
                            # Filter experiment results
                            exp_rounds = exp_attack_results[exp_attack_results['malicious_count'] == nbyz]['round_num'].unique()
                            
                            # Filter user scores by these rounds
                            nbyz_scores = user_scores[(user_scores['attack_type'] == attack_type) & 
                                                     (user_scores['experiment_id'] == selected_exp_id) &
                                                     (user_scores['round'].isin(exp_rounds))]
                            
                            if not nbyz_scores.empty:
                                # Find the closest round to the selected round
                                closest_round = nbyz_scores['round'].unique()[-1]  # Use the last round as the representative
                                round_nbyz_scores = nbyz_scores[nbyz_scores['round'] == closest_round]
                                
                                # Calculate metrics
                                benign_avg = round_nbyz_scores[round_nbyz_scores['is_actually_malicious'] == 0]['score'].mean()
                                malicious_avg = round_nbyz_scores[round_nbyz_scores['is_actually_malicious'] == 1]['score'].mean()
                                total_malicious = round_nbyz_scores[round_nbyz_scores['is_actually_malicious'] == 1].shape[0]
                                correctly_filtered = round_nbyz_scores[(round_nbyz_scores['is_actually_malicious'] == 1) & 
                                                                     (round_nbyz_scores['is_filtered_out'] == 1)].shape[0]
                                recall = correctly_filtered / total_malicious if total_malicious > 0 else 0
                                
                                nbyz_comparison.append({
                                    'Malicious Users': nbyz,
                                    'Benign Avg Score': benign_avg,
                                    'Malicious Avg Score': malicious_avg,
                                    'Score Gap': benign_avg - malicious_avg,
                                    'Recall': recall,
                                    'Round': closest_round
                                })
                        
                        if nbyz_comparison:
                            nbyz_df = pd.DataFrame(nbyz_comparison)
                            
                            # Display the table
                            st.dataframe(nbyz_df.set_index('Malicious Users'))
                            
                            # Create comparison charts
                            fig1 = px.bar(
                                nbyz_df,
                                x='Malicious Users',
                                y=['Benign Avg Score', 'Malicious Avg Score'],
                                barmode='group',
                                title='Average Scores by Number of Malicious Users',
                                labels={
                                    'value': 'Average Score',
                                    'variable': 'User Type'
                                }
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            fig2 = px.bar(
                                nbyz_df,
                                x='Malicious Users',
                                y='Score Gap',
                                title='Score Gap by Number of Malicious Users',
                                labels={
                                    'Score Gap': 'Score Gap (Benign - Malicious)'
                                },
                                color='Score Gap',
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            fig3 = px.bar(
                                nbyz_df,
                                x='Malicious Users',
                                y='Recall',
                                title='Detection Recall by Number of Malicious Users',
                                labels={
                                    'Recall': 'Recall'
                                },
                                color='Recall',
                                color_continuous_scale='Greens'
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("Only one value for number of malicious users is available.")
                
                with comp_tab2:
                    if len(bias_values) > 1:
                        st.write("### Comparison by Bias Value")
                        
                        # Create a comparison dataframe for the selected round across bias values
                        bias_comparison = []
                        
                        for bias in bias_values:
                            # Filter experiment results
                            if selected_nbyz is not None:
                                exp_rounds = exp_attack_results[(exp_attack_results['bias_values'] == bias) & 
                                                              (exp_attack_results['malicious_count'] == selected_nbyz)]['round_num'].unique()
                            else:
                                exp_rounds = exp_attack_results[exp_attack_results['bias_values'] == bias]['round_num'].unique()
                            
                            # Filter user scores by these rounds
                            bias_scores = user_scores[(user_scores['attack_type'] == attack_type) & 
                                                    (user_scores['experiment_id'] == selected_exp_id) &
                                                    (user_scores['round'].isin(exp_rounds))]
                            
                            if not bias_scores.empty:
                                # Find the closest round to the selected round
                                closest_round = bias_scores['round'].unique()[-1]  # Use the last round as the representative
                                round_bias_scores = bias_scores[bias_scores['round'] == closest_round]
                                
                                # Calculate metrics
                                benign_avg = round_bias_scores[round_bias_scores['is_actually_malicious'] == 0]['score'].mean()
                                malicious_avg = round_bias_scores[round_bias_scores['is_actually_malicious'] == 1]['score'].mean()
                                total_malicious = round_bias_scores[round_bias_scores['is_actually_malicious'] == 1].shape[0]
                                correctly_filtered = round_bias_scores[(round_bias_scores['is_actually_malicious'] == 1) & 
                                                                     (round_bias_scores['is_filtered_out'] == 1)].shape[0]
                                recall = correctly_filtered / total_malicious if total_malicious > 0 else 0
                                
                                bias_comparison.append({
                                    'Bias Value': bias,
                                    'Benign Avg Score': benign_avg,
                                    'Malicious Avg Score': malicious_avg,
                                    'Score Gap': benign_avg - malicious_avg,
                                    'Recall': recall,
                                    'Round': closest_round
                                })
                        
                        if bias_comparison:
                            bias_df = pd.DataFrame(bias_comparison)
                            
                            # Display the table
                            st.dataframe(bias_df.set_index('Bias Value'))
                            
                            # Create comparison charts
                            fig1 = px.bar(
                                bias_df,
                                x='Bias Value',
                                y=['Benign Avg Score', 'Malicious Avg Score'],
                                barmode='group',
                                title='Average Scores by Bias Value',
                                labels={
                                    'value': 'Average Score',
                                    'variable': 'User Type'
                                }
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            fig2 = px.bar(
                                bias_df,
                                x='Bias Value',
                                y='Score Gap',
                                title='Score Gap by Bias Value',
                                labels={
                                    'Score Gap': 'Score Gap (Benign - Malicious)'
                                },
                                color='Score Gap',
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            fig3 = px.bar(
                                bias_df,
                                x='Bias Value',
                                y='Recall',
                                title='Detection Recall by Bias Value',
                                labels={
                                    'Recall': 'Recall'
                                },
                                color='Recall',
                                color_continuous_scale='Greens'
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("Only one bias value is available.")
else:
    st.warning("No user score data available to visualize.")

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | User Scores Analysis")