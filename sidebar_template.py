import streamlit as st
import time

def create_sidebar():
    """
    Create a consistent sidebar across all pages

    Returns:
        None
    """
    st.sidebar.title("SafeFL Dashboard")
    st.sidebar.info("Hierarchical Federated Learning Visualization")

    # Experiment selector
    available_exp_ids = st.session_state.data_loader.get_available_experiment_ids()
    if available_exp_ids:
        current_exp_id = st.session_state.data_loader.current_experiment_id
        selected_exp_id = st.sidebar.selectbox(
            "Select Experiment ID",
            options=available_exp_ids,
            index=available_exp_ids.index(current_exp_id) if current_exp_id in available_exp_ids else -1,
            format_func=lambda x: f"Experiment {x}",
            help="The main experiment ID for all aggregation rules except HierarchicalFL"
        )

        # Update current experiment ID if changed
        if selected_exp_id != st.session_state.data_loader.current_experiment_id:
            st.session_state.data_loader.set_current_experiment_id(selected_exp_id)
            st.sidebar.success(f"Switched to Experiment {selected_exp_id}")
            st.experimental_rerun()

        # HierarchicalFL experiment selector
        st.sidebar.markdown("---")
        st.sidebar.subheader("HierarchicalFL Settings")

        # Option to use separate experiment ID for HierarchicalFL
        use_separate = st.sidebar.checkbox(
            "Use separate experiment for HierarchicalFL",
            value=st.session_state.data_loader.use_separate_hierarchical,
            help="When enabled, HierarchicalFL data will be loaded from a different experiment"
        )

        # If using separate experiment ID, show selector
        if use_separate:
            hier_exp_id = st.session_state.data_loader.hierarchical_experiment_id or current_exp_id
            hier_selected_exp_id = st.sidebar.selectbox(
                "HierarchicalFL Experiment ID",
                options=available_exp_ids,
                index=available_exp_ids.index(hier_exp_id) if hier_exp_id in available_exp_ids else -1,
                format_func=lambda x: f"Experiment {x}",
                help="The experiment ID specifically for HierarchicalFL aggregation rule"
            )

            # Update hierarchical experiment ID if changed
            if hier_selected_exp_id != st.session_state.data_loader.hierarchical_experiment_id:
                st.session_state.data_loader.set_hierarchical_experiment_id(hier_selected_exp_id)
                st.sidebar.success(f"HierarchicalFL will use data from Experiment {hier_selected_exp_id}")
                st.experimental_rerun()

        # Handle toggling separate experiment mode
        if use_separate != st.session_state.data_loader.use_separate_hierarchical:
            if use_separate:
                # Enable separate mode
                if st.session_state.data_loader.hierarchical_experiment_id is None:
                    # Default to same as current if not previously set
                    st.session_state.data_loader.set_hierarchical_experiment_id(current_exp_id)
                else:
                    # Just enable the mode
                    st.session_state.data_loader.use_separate_hierarchical = True
            else:
                # Disable separate mode
                st.session_state.data_loader.use_same_experiment_id()

            st.experimental_rerun()

        st.sidebar.markdown("---")

    # Data refresh control
    refresh_interval = st.sidebar.slider(
        "Auto-refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=15
    )

    if st.sidebar.button("Refresh Data Now"):
        st.session_state.data_loader.load_all_data()
        st.session_state.last_update_time = time.time()
        st.sidebar.success("Data refreshed!")

    # Auto-refresh logic
    if 'last_update_time' in st.session_state:
        current_time = time.time()
        if current_time - st.session_state.last_update_time > refresh_interval:
            # Check if files have changed
            if st.session_state.file_watcher.check_for_changes():
                st.session_state.data_loader.load_all_data()
                st.session_state.last_update_time = current_time

    # Display last update time
    if 'last_update_time' in st.session_state:
        st.sidebar.markdown(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_update_time))}")