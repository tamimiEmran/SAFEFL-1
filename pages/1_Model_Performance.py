import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sidebar_template import create_sidebar

# Set page configuration
st.set_page_config(
    page_title="Aggregation Comparison - Hierarchical Federated Learning",
    page_icon="⚖️",
    layout="wide"
)

# Ensure data loader is available
if 'data_loader' not in st.session_state:
    st.error("Please navigate to the main page first to initialize the data loader.")
    st.stop()

# Page title
st.title("Aggregation Rule Comparison")
st.markdown("Compare performance of different aggregation rules across bias values and attack types")

# Display current experiment ID
current_exp_id = st.session_state.data_loader.current_experiment_id
if current_exp_id is not None:
    st.info(f"Currently viewing Experiment ID: {current_exp_id}")

# Create consistent sidebar
create_sidebar()

# Get use_max_accuracy setting from session state
use_max_accuracy = st.session_state.get('use_max_accuracy', False)

# Display accuracy mode
accuracy_type = "maximum" if use_max_accuracy else "final round"
st.info(f"Showing {accuracy_type} accuracy values based on main dashboard setting")

# Get experiment results
experiment_results = st.session_state.data_loader.get_experiment_results()

if experiment_results is not None and not experiment_results.empty:
    # Extract unique values
    attack_types = sorted(experiment_results['malicious_type'].unique())
    bias_values = sorted(experiment_results['bias_values'].unique())
    protocols = sorted(experiment_results['aggregation_name'].unique())
    
    if len(attack_types) == 0:
        st.warning("No attack types found in the data.")
        st.stop()
    
    # Create tabs for different attack types
    attack_tabs = st.tabs([f"{attack_type}" for attack_type in attack_types])
    
    for i, attack_type in enumerate(attack_types):
        with attack_tabs[i]:
            st.subheader(f"Performance Analysis: {attack_type}")
            
            # Filter data for this attack type
            attack_data = experiment_results[experiment_results['malicious_type'] == attack_type]
            
            if attack_data.empty:
                st.warning(f"No data available for {attack_type}.")
                continue
            
            # Add bias value selector with a unique key for each attack type
            attack_bias_values = sorted(attack_data['bias_values'].unique())
            selected_bias = st.selectbox(
                "Select Bias Value",
                options=attack_bias_values,
                index=0,
                key=f"bias_selector_{attack_type}"  # Adding a unique key using the attack type
            )
            
            # Filter data for selected bias value
            bias_data = attack_data[attack_data['bias_values'] == selected_bias]
            
            # Find data to display for each protocol (latest round or max accuracy round)
            selected_rounds = []

            if use_max_accuracy:
                # For each protocol, find the round with max accuracy
                for protocol in protocols:
                    protocol_data = bias_data[bias_data['aggregation_name'] == protocol]
                    if not protocol_data.empty:
                        # Find round with maximum accuracy
                        max_acc_idx = protocol_data['round_accuracy'].idxmax()
                        best_round = protocol_data.loc[max_acc_idx]
                        selected_rounds.append(best_round)

                if selected_rounds:
                    st.caption("Showing maximum accuracy achieved for each protocol (may be from different rounds)")
            else:
                # Use the latest round for each protocol (original behavior)
                for protocol in protocols:
                    protocol_data = bias_data[bias_data['aggregation_name'] == protocol]
                    if not protocol_data.empty:
                        latest_round = protocol_data.loc[protocol_data['round_num'].idxmax()]
                        selected_rounds.append(latest_round)

                if selected_rounds:
                    st.caption("Showing accuracy from the final round for each protocol")

            if not selected_rounds:
                st.warning(f"No data available for {attack_type} with bias value {selected_bias}.")
                continue

            # Create a DataFrame for the results
            results_df = pd.DataFrame(selected_rounds)
            
            # Select only relevant columns
            display_columns = ['aggregation_name', 'round_num', 'round_accuracy']
            
            # Add backdoor success rate for scaling attacks
            if 'round_backdoor_success' in results_df.columns:
                display_columns.append('round_backdoor_success')
            
            # Rename columns for better display
            results_df = results_df[display_columns].rename(columns={
                'aggregation_name': 'Aggregation Rule',
                'round_num': 'Round',
                'round_accuracy': 'Model Accuracy',
                'round_backdoor_success': 'Backdoor Success Rate'
            })

            # Add note about round information
            if use_max_accuracy:
                results_df['Description'] = results_df.apply(
                    lambda row: f"Max accuracy achieved in round {int(row['Round'])}", axis=1
                )
            else:
                results_df['Description'] = 'Final round'
            
            # Sort by accuracy (descending)
            results_df = results_df.sort_values('Model Accuracy', ascending=False)
            
            # Format accuracy and backdoor values as percentages
            results_df['Model Accuracy'] = results_df['Model Accuracy'].apply(lambda x: f"{x:.2%}")
            if 'Backdoor Success Rate' in results_df.columns:
                results_df['Backdoor Success Rate'] = results_df['Backdoor Success Rate'].apply(lambda x: f"{x:.2%}")
            
            # Display the table
            st.subheader(f"Results for Bias Value: {selected_bias}")
            st.dataframe(results_df, use_container_width=True)
            
            # Highlight the best performing aggregation rule with improvement over second best
            if len(results_df) > 1:
                # Convert accuracy percentages back to floats for comparison
                accuracy_values = results_df['Model Accuracy'].apply(
                    lambda x: float(x.strip('%').replace(',', '')) / 100 if isinstance(x, str) else x
                )
                
                # Get the top two performers
                best_accuracy = accuracy_values.iloc[0]
                second_best_accuracy = accuracy_values.iloc[1]
                
                # Calculate improvement
                improvement = best_accuracy - second_best_accuracy
                improvement_percentage = improvement * 100
                
                # Get the best rule name
                best_rule = results_df.iloc[0]['Aggregation Rule']
                second_best_rule = results_df.iloc[1]['Aggregation Rule']
                
                # Display with improvement
                st.success(f"Best performing aggregation rule for model accuracy: **{best_rule}** with a **{improvement_percentage:.2f}%** improvement over {second_best_rule}")
            else:
                # Only one aggregation rule available
                best_rule = results_df.iloc[0]['Aggregation Rule']
                st.success(f"Best performing aggregation rule for model accuracy: **{best_rule}**")
            
            # If backdoor data is available AND it's a scaling attack, highlight the best for backdoor resistance
            if 'Backdoor Success Rate' in results_df.columns and attack_type == 'scaling_attack':
                # Create a copy and convert percentage strings back to float for proper sorting
                backdoor_df = results_df.copy()
                backdoor_df['Backdoor Value'] = backdoor_df['Backdoor Success Rate'].apply(
                    lambda x: float(x.strip('%').replace(',', '')) / 100 if isinstance(x, str) else x
                )
                
                # Sort by backdoor success (ascending, lower is better)
                backdoor_df = backdoor_df.sort_values('Backdoor Value', ascending=True)
                
                # Highlight the best for backdoor resistance with improvement over second best
                if len(backdoor_df) > 1:
                    # Get the top two performers
                    best_backdoor_value = backdoor_df['Backdoor Value'].iloc[0]
                    second_best_backdoor_value = backdoor_df['Backdoor Value'].iloc[1]
                    
                    # Calculate improvement (for backdoor, lower is better, so we subtract in reverse order)
                    improvement = second_best_backdoor_value - best_backdoor_value
                    improvement_percentage = improvement * 100
                    
                    # Get the rule names
                    best_backdoor_rule = backdoor_df.iloc[0]['Aggregation Rule']
                    second_best_backdoor_rule = backdoor_df.iloc[1]['Aggregation Rule']
                    
                    # Display with improvement
                    st.success(f"Best performing aggregation rule for backdoor resistance: **{best_backdoor_rule}** with a **{improvement_percentage:.2f}%** improvement over {second_best_backdoor_rule}")
                else:
                    # Only one aggregation rule available
                    best_backdoor_rule = backdoor_df.iloc[0]['Aggregation Rule']
                    st.success(f"Best performing aggregation rule for backdoor resistance: **{best_backdoor_rule}**")
else:
    st.warning("No experiment results available to compare aggregation rules.")

# Footer
st.markdown("---")
st.markdown("Hierarchical Federated Learning Dashboard | Aggregation Rule Comparison")