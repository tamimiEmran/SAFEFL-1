import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Scoring Function Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Title and Introduction ---
st.title("🛡️ Analysis of Group-Based Scoring Functions")
st.markdown("""
This dashboard analyzes the effectiveness of different scoring functions in identifying malicious groups.
Use the controls on the left to select a **dataset**, **attack scenario**, **scoring function**, and **group size** to explore.
""")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath):
    """Loads the results CSV and performs initial processing."""
    try:
        df = pd.read_csv(filepath)
        # 2. Create a simple 'Benign' vs 'Malicious' category for easier plotting.
        df['group_type'] = df['group_id_mal_size'].apply(lambda x: 'Benign' if x == 0 else 'Malicious')
        
        return df
    except FileNotFoundError:
        # [IMPROVEMENT] The error message is now dynamic.
        st.error(f"Error: The data file was not found at '{filepath}'. Please make sure it is in the same directory.")
        return None

# The normalize_score function is no longer needed as we will handle the logic directly
# in the plot titles for clarity.

# --- Main App ---

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")

# =======================================================================
# --- NEW FEATURE: Dataset Selection ---
# First, select the dataset (MNIST or HAR)
dataset_type = st.sidebar.radio(
    "1. Select Dataset",
    options=["MNIST", "HAR"],
    help="Choose between MNIST and HAR datasets."
)

# Create a mapping from user-friendly names to filenames based on dataset selection.
if dataset_type == "MNIST":
    DATASET_OPTIONS = {
        "Scaling Attack": "final.csv",
        "Label-Flipping Attack": "HAR_final_fixed_f.csv"
    }
else:  # HAR
    DATASET_OPTIONS = {
        "Scaling Attack": "HAR_final_fixed_f.csv",
        "Label-Flipping Attack": "HAR_final_fixed_f.csv"
    }

# Add a radio button to the sidebar to select the attack scenario.
selected_dataset_name = st.sidebar.radio(
    "2. Select Attack Scenario",
    options=DATASET_OPTIONS.keys(),
    help="Choose the attack type to analyze the defenses against."
)

# Get the corresponding filename.
selected_filepath = DATASET_OPTIONS[selected_dataset_name]
# =======================================================================

# Load the selected data file.
df = load_data(selected_filepath)

if df is not None:
    # --- Continue with the rest of the sidebar controls ---
    
    # Get unique values for the select boxes from the *currently loaded* dataframe.
    available_scorers = df['scoring_func'].unique()
    available_group_sizes = sorted(df['group_size'].unique())

    # UI Element to select the scoring function
    selected_scorer = st.sidebar.selectbox(
        "3. Select Scoring Function",
        options=available_scorers,
        index=0, # Default to the first scorer
        help="Choose the defense algorithm score you want to analyze."
    )

    # UI Element to select the group size
    selected_group_size = st.sidebar.selectbox(
        "4. Select Group Size",
        options=available_group_sizes,
        index=0, # Default to the first group size
        help="Analyze the performance for a specific group size."
    )

    # UI Element to select the grouping aggregation method
    grouping_agg_options = df['grouping_agg'].unique()
    if len(grouping_agg_options) > 0:
        selected_grouping_agg = st.sidebar.selectbox(
            "5. Select Grouping Aggregation Method",
            options=grouping_agg_options,
            index=0,  # Default to the first option
            help="Choose the aggregation method used for grouping clients."
        )

    # --- Filter Data based on User Selection ---
    filtered_df = df[
        (df['scoring_func'] == selected_scorer) &
        (df['group_size'] == selected_group_size) &
        (df['grouping_agg'] == selected_grouping_agg)
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected combination.")
    else:
        # --- Display Dataset Information ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Dataset Information")
        st.sidebar.markdown(f"**Dataset:** {dataset_type}")
        st.sidebar.markdown(f"**Attack:** {selected_dataset_name}")
        st.sidebar.markdown(f"**Total Records:** {len(filtered_df)}")
        st.sidebar.markdown(f"**Unique Groups:** {filtered_df['group_id'].nunique()}")
        st.sidebar.markdown(f"**Unique Clients:** {filtered_df['client_id'].nunique()}")
        # --- Visualization 1: Score Distribution by Maliciousness Level (Box Plot) ---
        st.header("1. Score Distribution by Group Maliciousness")
        st.markdown(f"""
        This plot shows the distribution of scores for groups based on the **exact number of malicious members** they contain.
        **A good scorer should show a clear trend**: as the number of malicious members increases, the scores should consistently increase or decrease.
        """)

        # Determine the direction for the title dynamically for clarity.
        is_dnc_scorer = 'dnc_bb_prob_benign' in selected_scorer
        # For D&C, score is prob_benign, so LOW score = MALICIOUS.
        # For others, a LOW score = MALICIOUS.
        score_direction_text = "(Lower Score = More Malicious)"

        fig_box = px.box(
            filtered_df,
            x='group_id_mal_size',
            y='score',
            color='group_id_mal_size',
            points="all",
            title=f"<b>{selected_scorer}</b> Scores for Group Size {selected_group_size}<br><sup>{score_direction_text}</sup>",
            labels={
                "group_id_mal_size": "Number of Malicious Members in Group",
                "score": "Score"
            }
        )
        fig_box.update_layout(xaxis_title_font_size=16, yaxis_title_font_size=16)
        st.plotly_chart(fig_box, use_container_width=True)


        # --- Visualization 2: Score Separability (Overlaid Histograms) ---
        st.header("2. Score Separability: Benign vs. Malicious Groups")
        st.markdown("""
        This plot compares the score distribution of purely **Benign** groups (0 malicious members)
        against all **Malicious** groups (>0 malicious members).
        **A good scorer shows two distinct peaks with minimal overlap**, making it easy to set a threshold to separate them.
        """)

        fig_hist = px.histogram(
            filtered_df,
            x="score",
            color="group_type",
            marginal="box",
            histnorm='percent',
            barmode='overlay',
            opacity=0.7,
            title=f"<b>{selected_scorer}</b> Score Distribution: Benign vs. Malicious Groups<br><sup>{score_direction_text}</sup>",
            labels={
                "score": "Score",
                "group_type": "Group Type"
            }
        )
        fig_hist.update_layout(xaxis_title_font_size=16, yaxis_title_font_size=16, legend_title_font_size=14)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # --- Additional Dataset-Specific Visualizations ---
        st.header("3. Dataset-Specific Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Malicious vs Benign Users Distribution
            mal_dist = filtered_df.groupby('is_user_malicious').size().reset_index(name='count')
            mal_dist['is_user_malicious'] = mal_dist['is_user_malicious'].map({True: 'Malicious', False: 'Benign'})
            
            fig_mal_dist = px.pie(
                mal_dist,
                values='count',
                names='is_user_malicious',
                title=f"User Distribution in {dataset_type} Dataset",
                color_discrete_map={'Malicious': '#FF6B6B', 'Benign': '#4ECDC4'}
            )
            st.plotly_chart(fig_mal_dist, use_container_width=True)
        
        with col2:
            # Group Maliciousness Distribution
            group_mal_dist = filtered_df.groupby('group_id_mal_size').size().reset_index(name='count')
            
            fig_group_mal = px.bar(
                group_mal_dist,
                x='group_id_mal_size',
                y='count',
                title="Distribution of Malicious Users per Group",
                labels={
                    "group_id_mal_size": "Number of Malicious Users in Group",
                    "count": "Number of Groups"
                },
                color='group_id_mal_size',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_group_mal, use_container_width=True)
        
        # --- Score Effectiveness Analysis ---
        st.header("4. Scoring Function Effectiveness")
        
        # Calculate metrics for effectiveness
        benign_scores = filtered_df[filtered_df['group_id_mal_size'] == 0]['score']
        malicious_scores = filtered_df[filtered_df['group_id_mal_size'] > 0]['score']
        
        if len(benign_scores) > 0 and len(malicious_scores) > 0:
            # Calculate separation metrics
            benign_mean = benign_scores.mean()
            malicious_mean = malicious_scores.mean()
            benign_std = benign_scores.std()
            malicious_std = malicious_scores.std()
            
            # Cohen's d for effect size
            pooled_std = ((len(benign_scores) - 1) * benign_std**2 + (len(malicious_scores) - 1) * malicious_std**2) / (len(benign_scores) + len(malicious_scores) - 2)
            pooled_std = pooled_std**0.5
            cohens_d = abs(benign_mean - malicious_mean) / pooled_std if pooled_std > 0 else 0
            
            # Display metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Benign Mean Score", f"{benign_mean:.4f}")
                st.metric("Benign Std Dev", f"{benign_std:.4f}")
            
            with metric_col2:
                st.metric("Malicious Mean Score", f"{malicious_mean:.4f}")
                st.metric("Malicious Std Dev", f"{malicious_std:.4f}")
            
            with metric_col3:
                st.metric("Score Separation (Cohen's d)", f"{cohens_d:.4f}")
                if cohens_d < 0.2:
                    effect_size = "Negligible"
                elif cohens_d < 0.5:
                    effect_size = "Small"
                elif cohens_d < 0.8:
                    effect_size = "Medium"
                else:
                    effect_size = "Large"
                st.metric("Effect Size", effect_size)
        
        # --- Round-based Analysis if available ---
        if 'round_id' in filtered_df.columns:
            st.header("5. Score Evolution Across Rounds")
            
            # Average score by round and maliciousness
            round_scores = filtered_df.groupby(['round_id', 'group_type'])['score'].mean().reset_index()
            
            fig_rounds = px.line(
                round_scores,
                x='round_id',
                y='score',
                color='group_type',
                title=f"Average {selected_scorer} Score Evolution",
                labels={
                    "round_id": "Round",
                    "score": "Average Score",
                    "group_type": "Group Type"
                },
                markers=True
            )
            st.plotly_chart(fig_rounds, use_container_width=True)