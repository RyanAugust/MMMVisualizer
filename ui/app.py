import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://0.0.0.0:8000"

# Google Brand Colors
G_BLUE = "#4285F4"
G_RED = "#DB4437"
G_YELLOW = "#F4B400"
G_GREEN = "#0F9D58"
GOOGLE_PALETTE = [G_BLUE, G_RED, G_YELLOW, G_GREEN, "#AB47BC", "#00ACC1", "#FF7043"]

st.set_page_config(page_title="BikeShop MMM Director", layout="wide")

st.title("🚲 BikeShop Marketing Mix Model Director")
st.markdown("---")

# Helper for visualizations
def plot_adstock_curve(lambda_val):
    days = np.arange(0, 21)
    decay = lambda_val ** days
    fig = px.line(x=days, y=decay, title="Geometric Adstock Decay",
                 labels={'x': 'Days After Spend', 'y': 'Retention Factor'},
                 color_discrete_sequence=[G_BLUE])
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_saturation_curve(alpha, gamma):
    spend = np.linspace(0, gamma * 5, 100)
    response = (spend**alpha) / (spend**alpha + gamma**alpha)
    fig = px.line(x=spend, y=response, title="Hill Saturation Curve",
                 labels={'x': 'Weekly Spend ($)', 'y': 'Relative Response'},
                 color_discrete_sequence=[G_RED])
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Load current config globally
try:
    response = requests.get(f"{API_URL}/api/config")
    if response.status_code == 200:
        config = response.json()
        channels = config.get("channels", [])
    else:
        st.error("Failed to load configuration from API.")
        st.stop()
except Exception as e:
    st.error(f"Could not connect to API: {e}")
    st.stop()

# Navigation
tabs = st.tabs(["Stage 1: Model Configuration", "Stage 2: Investment Simulator", "Stage 3: Data Explorer", "Stage 4: Marginal Efficiency", "Stage 5: Reach & Frequency"])

with tabs[0]:
    st.header("Stage 1: Configure Marketing Landscape")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Basic Parameters")
        years = st.number_input("Years of Data", value=config["basic"].get("years", 3), min_value=1, max_value=5)
        start_date = st.text_input("Start Date (yyyy/mm/dd)", value=config["basic"].get("start_date", "2022/01/01"))
        freq = st.number_input("Campaign Frequency (Days)", value=config["basic"].get("frequency_of_campaigns", 7), min_value=1)
        revenue_per_conv = st.number_input("Revenue per Conversion ($)", value=config["basic"].get("revenue_per_conv", 500.0))
        
        st.subheader("Baseline Parameters")
        bl_cfg = config["baseline"]
        base_p = st.number_input("Daily Base Sales (Units)", value=bl_cfg["base_p"])
        trend_p = st.number_input("Total Growth Trend (Units)", value=bl_cfg["trend_p"])
        error_std = st.number_input("Baseline Noise (Error Std)", value=bl_cfg.get("error_std", 50), min_value=0)
        
        st.subheader("Seasonality")
        temp_var = st.number_input("Seasonality Variance (Height)", value=bl_cfg["temp_var"])
        temp_mean = st.slider("Seasonality Importance (Mean)", 0.0, 1.0, value=bl_cfg.get("temp_coef_mean", 0.5), step=0.05)
        temp_sd = st.slider("Seasonality Variability (SD)", 0.0, 0.5, value=bl_cfg.get("temp_coef_sd", 0.1), step=0.01)
        
    with col2:
        st.subheader("Channels")
        current_channels = config.get("channels", [])
        
        # Form to add a channel
        with st.expander("Add New Channel", expanded=not bool(current_channels)):
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                new_channel_name = st.text_input("Channel Name")
                new_channel_type = st.selectbox("Channel Type", ["Impressions", "Clicks", "Reach & Frequency"])
                new_channel_cvr = st.slider("True CVR", 0.0, 0.1, 0.01, step=0.001, format="%.3f")
                
                cost_label = "True CPM ($)" if new_channel_type in ["Impressions", "Reach & Frequency"] else "True CPC ($)"
                new_channel_cost = st.number_input(cost_label, value=10.0 if new_channel_type in ["Impressions", "Reach & Frequency"] else 1.0)
                new_avg_spend = st.number_input("Avg Weekly Spend ($)", value=5000)
                
            with c_col2:
                st.write("**Modeling Parameters**")
                new_lambda = st.slider("Adstock Lambda (Geometric Decay)", 0.0, 1.0, 0.5, step=0.05)
                new_alpha = st.slider("Saturation Alpha (Hill Shape)", 0.5, 3.0, 1.0, step=0.1)
                new_gamma = st.number_input("Saturation Gamma (Half Sat Point $)", value=5000)
                
                st.write("**Noise Parameters**")
                new_cost_noise = st.slider("Cost Noise (Scale)", 0.0, 0.2, 0.05, step=0.01)
                new_cvr_noise = st.slider("CVR Noise (Scale)", 0.0, 0.2, 0.05, step=0.01)

            v_col1, v_col2 = st.columns(2)
            v_col1.plotly_chart(plot_adstock_curve(new_lambda), width="stretch", key="new_channel_adstock")
            v_col2.plotly_chart(plot_saturation_curve(new_alpha, new_gamma), width="stretch", key="new_channel_saturation")
            
            if st.button("Add Channel"):
                if new_channel_name:
                    rf_params = None
                    if new_channel_type == "Reach & Frequency":
                        rf_params = {"max_reach": 0.8, "reach_slope": 1.0}
                    
                    current_channels.append({
                        "name": new_channel_name,
                        "type": new_channel_type,
                        "true_cvr": new_channel_cvr,
                        "true_cost": float(new_channel_cost),
                        "avg_spend": float(new_avg_spend),
                        "adstock": {"type": "geometric", "params": {"lambda_": new_lambda}},
                        "saturation": {"type": "hill", "params": {"alpha": new_alpha, "gamma": float(new_gamma)}},
                        "cost_noise": {"loc": 0.0, "scale": new_cost_noise},
                        "cvr_noise": {"loc": 0.0, "scale": new_cvr_noise},
                        "rf_params": rf_params
                    })
                    config["channels"] = current_channels
                    requests.post(f"{API_URL}/api/config", json=config)
                    st.rerun()

        # Display and remove channels
        if current_channels:
            st.markdown("---")
            for i, channel in enumerate(current_channels):
                with st.container():
                    d_col1, d_col2, d_col3 = st.columns([2, 3, 1])
                    d_col1.write(f"### {channel['name']}")
                    d_col1.write(f"**Type:** {channel['type']}")
                    d_col1.write(f"**True CVR:** {channel['true_cvr']:.3f}")
                    cost_unit = "CPM" if channel['type'] == "Impressions" else "CPC"
                    d_col1.write(f"**Cost ({cost_unit}):** ${channel['true_cost']:.2f}")
                    d_col1.write(f"**Avg Weekly Spend:** ${channel.get('avg_spend', 5000):,.0f}")
                    
                    with d_col2:
                        dv_col1, dv_col2 = st.columns(2)
                        lambda_val = channel.get("adstock", {}).get("params", {}).get("lambda_", 0.5)
                        alpha = channel.get("saturation", {}).get("params", {}).get("alpha", 1.0)
                        gamma = channel.get("saturation", {}).get("params", {}).get("gamma", 5000.0)
                        
                        fig_a = plot_adstock_curve(lambda_val)
                        fig_s = plot_saturation_curve(alpha, gamma)
                        fig_a.update_layout(height=150, title="Adstock")
                        fig_s.update_layout(height=150, title="Saturation")
                        dv_col1.plotly_chart(fig_a, width="stretch", key=f"list_adstock_{i}_{channel['name']}")
                        dv_col2.plotly_chart(fig_s, width="stretch", key=f"list_saturation_{i}_{channel['name']}")

                    if d_col3.button("Remove", key=f"remove_{i}"):
                        current_channels.pop(i)
                        config["channels"] = current_channels
                        requests.post(f"{API_URL}/api/config", json=config)
                        st.rerun()
                st.markdown("---")

    if st.button("Save & Train Model", type="primary"):
        # Update config with current inputs
        config["basic"]["years"] = years
        config["basic"]["start_date"] = start_date
        config["basic"]["frequency_of_campaigns"] = freq
        config["basic"]["revenue_per_conv"] = revenue_per_conv
        
        config["baseline"]["base_p"] = base_p
        config["baseline"]["trend_p"] = trend_p
        config["baseline"]["error_std"] = error_std
        config["baseline"]["temp_var"] = temp_var
        config["baseline"]["temp_coef_mean"] = temp_mean
        config["baseline"]["temp_coef_sd"] = temp_sd
        
        requests.post(f"{API_URL}/api/config", json=config)
        st.info("ℹ️ Training actual Google Meridian Bayesian model. This involves MCMC sampling and may take a few minutes...")
        with st.spinner("Executing Meridian MCMC Sampler..."):
            train_response = requests.post(f"{API_URL}/api/train")
            if train_response.status_code == 200:
                st.success("Meridian Model trained successfully with posterior parameters!")
            else:
                st.error(f"Training failed: {train_response.text}")

with tabs[1]:
    st.header("Stage 2: Investment Simulator")
    
    if not channels:
        st.warning("Please configure channels in Stage 1 first.")
    else:
        # Default investment to the sum of average weekly spends from Stage 1
        total_investment_default = sum([c.get("avg_spend", 5000.0) for c in channels])
        
        st.subheader("Budget Strategy")
        total_investment = st.number_input("Total Weekly Marketing Investment ($)", 
                                         value=float(total_investment_default), 
                                         step=1000.0)
        
        # Initialize session state for allocations and locks
        if "allocations" not in st.session_state:
            st.session_state.allocations = {}
            st.session_state.locks = {}
            for channel in channels:
                st.session_state.allocations[channel['name']] = int(100 / len(channels))
                st.session_state.locks[channel['name']] = False

        # Optimization Button
        if st.button("🚀 Optimize Allocation", type="secondary"):
            fixed_data = {}
            for c_name, is_locked in st.session_state.locks.items():
                if is_locked:
                    fixed_data[c_name] = (st.session_state.allocations[c_name] / 100) * total_investment

            with st.spinner("Calculating optimal mix..."):
                opt_res = requests.post(f"{API_URL}/api/optimize", 
                                      json={"total_budget": total_investment, "fixed_allocations": fixed_data})
                if opt_res.status_code == 200:
                    opt_data = opt_res.json()
                    if opt_data["status"] == "success":
                        for c_name, data in opt_data["allocations"].items():
                            if not st.session_state.locks.get(c_name):
                                st.session_state.allocations[c_name] = int(data["percentage"])
                        st.success("Allocation optimized!")
                        st.rerun()
                    else:
                        st.error(f"Optimization failed: {opt_data.get('message')}")

        st.markdown("---")
        st.subheader("Channel Allocation (%)")
        
        allocation = {}
        cols = st.columns(len(channels))
        total_alloc = 0
        for i, channel in enumerate(channels):
            with cols[i]:
                is_locked = st.checkbox(f"Lock {channel['name']}", 
                                      value=st.session_state.locks.get(channel['name'], False),
                                      key=f"lock_{channel['name']}")
                st.session_state.locks[channel['name']] = is_locked
                
                percent = st.slider(f"{channel['name']} (%)", 0, 100, 
                                  value=st.session_state.allocations.get(channel['name'], 0),
                                  key=f"alloc_slider_{channel['name']}",
                                  disabled=is_locked)
                allocation[channel['name']] = percent
                st.session_state.allocations[channel['name']] = percent 
                total_alloc += percent
        
        if total_alloc > 100:
            st.error(f"Total allocation is {total_alloc}%. Please reduce to 100% or less.")
        elif total_alloc < 100:
            st.warning(f"Total allocation is {total_alloc}%. You have {100-total_alloc}% of budget unallocated.")

        spend_decisions = {}
        for channel_name, percent in allocation.items():
            weekly_spend = (percent / 100) * total_investment
            spend_decisions[channel_name] = weekly_spend / 7 
        
        st.write("**Calculated Weekly Spend:**")
        calc_cols = st.columns(len(channels))
        for i, channel in enumerate(channels):
            calc_cols[i].metric(channel['name'], f"${(allocation[channel['name']]/100 * total_investment):,.0f}")

        st.markdown("---")
        if total_alloc <= 100:
            try:
                predict_response = requests.post(f"{API_URL}/api/predict", json=spend_decisions)
                if predict_response.status_code == 200:
                    results = predict_response.json()
                    st.metric("Total Predicted Daily Revenue", f"${results['total_predicted_revenue']:,.2f}")
                    
                    breakdown_df = pd.DataFrame(results["channel_breakdown"])
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        fig = px.pie(breakdown_df, values='predicted_revenue', names='channel', title='Revenue Contribution by Channel',
                                    color_discrete_sequence=GOOGLE_PALETTE)
                        st.plotly_chart(fig, width="stretch", key="prediction_pie")
                    with v_col2:
                        fig2 = px.bar(breakdown_df, x='channel', y='predicted_revenue', title='Predicted Revenue per Channel',
                                     color_discrete_sequence=[G_BLUE])
                        st.plotly_chart(fig2, width="stretch", key="prediction_bar")

                    st.subheader("Saturation & Investment Alignment")
                    roi_cols = st.columns(min(len(channels), 3))
                    for i, channel in enumerate(channels):
                        col_idx = i % 3
                        with roi_cols[col_idx]:
                            sat_cfg = channel.get("saturation", {"type": "hill", "params": {"alpha": 1.0, "gamma": 5000.0}})
                            alpha = sat_cfg["params"].get("alpha", 1.0)
                            gamma = sat_cfg["params"].get("gamma", 5000.0)
                            
                            current_weekly_spend = spend_decisions[channel['name']] * 7
                            max_x = max(gamma * 2, current_weekly_spend * 1.5)
                            
                            # Split ranges: 0 to current, and current to max
                            spend_hist = np.linspace(0, current_weekly_spend, 50)
                            spend_proj = np.linspace(current_weekly_spend, max_x, 50)
                            
                            resp_hist = (spend_hist**alpha) / (spend_hist**alpha + gamma**alpha)
                            resp_proj = (spend_proj**alpha) / (spend_proj**alpha + gamma**alpha)
                            current_response = (current_weekly_spend**alpha) / (current_weekly_spend**alpha + gamma**alpha)
                            
                            fig_roi = go.Figure()
                            # Solid Line (Current/Historical)
                            fig_roi.add_trace(go.Scatter(x=spend_hist, y=resp_hist, name="Actual", line=dict(color=G_BLUE)))
                            # Dashed Line (Projected)
                            fig_roi.add_trace(go.Scatter(x=spend_proj, y=resp_proj, name="Projected", line=dict(color=G_BLUE, dash='dash')))
                            # Current Marker
                            fig_roi.add_trace(go.Scatter(x=[current_weekly_spend], y=[current_response], mode='markers', marker=dict(color=G_RED, size=12)))
                            fig_roi.update_layout(title=f"{channel['name']} Saturation", xaxis_title="Weekly Spend ($)", yaxis_title="Relative Response", height=300, showlegend=False)
                            st.plotly_chart(fig_roi, width="stretch", key=f"roi_curve_{i}_{channel['name']}")

                    st.subheader("Combined Channel Response Comparison")
                    fig_combined = go.Figure()
                    for i, channel in enumerate(channels):
                        sat_cfg = channel.get("saturation", {"type": "hill", "params": {"alpha": 1.0, "gamma": 5000.0}})
                        alpha = sat_cfg["params"].get("alpha", 1.0)
                        gamma = sat_cfg["params"].get("gamma", 5000.0)
                        cvr = channel['true_cvr']
                        rev_per_conv = config["basic"]['revenue_per_conv']
                        current_weekly_spend = spend_decisions[channel['name']] * 7
                        max_x = max(gamma * 2, current_weekly_spend * 1.5)
                        
                        # Split ranges
                        spend_hist = np.linspace(0, current_weekly_spend, 50)
                        spend_proj = np.linspace(current_weekly_spend, max_x, 50)
                        
                        max_weekly_rev = 10000 * 10 * cvr * rev_per_conv # Proxy max
                        
                        rev_hist = max_weekly_rev * ((spend_hist**alpha) / (spend_hist**alpha + gamma**alpha))
                        rev_proj = max_weekly_rev * ((spend_proj**alpha) / (spend_proj**alpha + gamma**alpha))
                        current_rev = max_weekly_rev * ((current_weekly_spend**alpha) / (current_weekly_spend**alpha + gamma**alpha))
                        
                        color = GOOGLE_PALETTE[i % len(GOOGLE_PALETTE)]
                        
                        # Solid line (Actual)
                        fig_combined.add_trace(go.Scatter(x=spend_hist, y=rev_hist, 
                                                       name=channel['name'], 
                                                       legendgroup=channel['name'],
                                                       line=dict(color=color), showlegend=True))
                        # Dashed line (Projected)
                        fig_combined.add_trace(go.Scatter(x=spend_proj, y=rev_proj, 
                                                       name=f"{channel['name']} (Proj)", 
                                                       legendgroup=channel['name'],
                                                       line=dict(color=color, dash='dash'), showlegend=False))
                        # The Marker
                        fig_combined.add_trace(go.Scatter(x=[current_weekly_spend], y=[current_rev], 
                                                       mode='markers', 
                                                       legendgroup=channel['name'],
                                                       showlegend=False,
                                                       marker=dict(color=color, size=12, symbol='diamond')))

                    fig_combined.update_layout(xaxis_title="Weekly Spend ($)", yaxis_title="Predicted Weekly Revenue ($)", height=500, hovermode="x unified")
                    st.plotly_chart(fig_combined, width="stretch", key="combined_response_chart")
                else:
                    st.warning("Please train the model in Stage 1 to see predictions.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tabs[2]:
    st.header("Stage 3: Data Explorer")
    if not channels:
        st.warning("Please configure channels in Stage 1 first.")
    else:
        if st.button("🔄 Fetch & Refresh Data"):
            with st.spinner("Generating data from PySiMMMulator..."):
                try:
                    data_res = requests.get(f"{API_URL}/api/data")
                    if data_res.status_code == 200:
                        raw_data = data_res.json()
                        df_viz = pd.DataFrame(raw_data)
                        df_viz['date'] = pd.to_datetime(df_viz['date'])
                        st.session_state.raw_df = df_viz
                        st.success("Data fetched successfully!")
                    else:
                        st.error(f"Failed to fetch data: {data_res.text}")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        if "raw_df" in st.session_state:
            df = st.session_state.raw_df
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                metric_to_plot = st.selectbox("Select Media Metric", options=[c for c in df.columns if "spend" in c or "impressions" in c or "clicks" in c])
            with col_f2:
                time_agg = st.radio("Time Aggregation", ["Daily", "Weekly"], horizontal=True)
            df_plot = df.set_index('date').resample('W').sum().reset_index() if time_agg == "Weekly" else df
            fig_explore = go.Figure()
            fig_explore.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[metric_to_plot], name=metric_to_plot.replace("_", " ").title(), line=dict(color=G_BLUE)))
            fig_explore.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['total_conversions'], name="Total Purchases", line=dict(color=G_YELLOW, dash='dot'), yaxis="y2"))
            fig_explore.update_layout(
                title=f"{metric_to_plot.replace('_', ' ').title()} vs Total Purchases Over Time",
                xaxis_title="Date",
                yaxis=dict(
                    title=dict(text=metric_to_plot.replace("_", " ").title(), font=dict(color=G_BLUE)),
                    tickfont=dict(color=G_BLUE)
                ),
                yaxis2=dict(
                    title=dict(text="Total Purchases", font=dict(color=G_YELLOW)),
                    tickfont=dict(color=G_YELLOW), 
                    overlaying="y", 
                    side="right"
                ),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_explore, width="stretch", key="explorer_main_chart")
            with st.expander("View Raw Data Table"):
                st.dataframe(df.head(100))
        else:
            st.info("Click 'Fetch & Refresh Data' to visualize the training data.")

with tabs[3]:
    st.header("Stage 4: Marginal Efficiency Analysis")
    st.markdown("Analyze how efficiency (CPA) and optimal budget allocation change as you scale your total marketing investment.")
    
    if not channels:
        st.warning("Please configure channels in Stage 1 first.")
    else:
        total_current_weekly = sum([c.get("avg_spend", 5000.0) for c in channels])
        
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            st.subheader("Analysis Settings")
            min_budget = st.number_input("Min Weekly Budget ($)", value=max(1000.0, total_current_weekly * 0.5), step=1000.0)
            max_budget = st.number_input("Max Weekly Budget ($)", value=total_current_weekly * 2.0, step=1000.0)
            steps = 100
            
            if min_budget >= max_budget:
                st.error("Max budget must be greater than min budget.")
                run_analysis = False
            else:
                run_analysis = st.button("📈 Run Efficiency Analysis", type="primary")

        if run_analysis:
            budgets = np.linspace(min_budget, max_budget, steps)
            efficiency_results = []
            allocation_history = []

            with st.spinner("Sweeping budget levels and optimizing..."):
                for budget in budgets:
                    opt_res = requests.post(f"{API_URL}/api/optimize", json={"total_budget": float(budget)})
                    if opt_res.status_code == 200:
                        data = opt_res.json()
                        if data["status"] == "success":
                            rev = data["expected_weekly_revenue"]
                            acqs = rev / config["basic"]["revenue_per_conv"]
                            cpa = budget / acqs if acqs > 0 else 0
                            
                            efficiency_results.append({
                                "Total Weekly Spend": budget,
                                "Total Predicted Revenue": rev,
                                "Total Predicted Acquisitions": acqs,
                                "CPA": cpa
                            })
                            
                            for c_name, alloc in data["allocations"].items():
                                allocation_history.append({
                                    "Total Weekly Spend": budget,
                                    "Channel": c_name,
                                    "Optimal Allocation ($)": alloc["weekly_spend"],
                                    "Optimal Allocation (%)": alloc["percentage"]
                                })

            if efficiency_results:
                df_eff = pd.DataFrame(efficiency_results)
                df_alloc = pd.DataFrame(allocation_history)

                # Plot 1: Total Spend vs CPA
                fig_cpa = px.line(df_eff, x="Total Weekly Spend", y="CPA", 
                                title="CPA Efficiency Curve",
                                color_discrete_sequence=[G_RED])
                fig_cpa.update_layout(yaxis_title="Cost Per Acquisition ($)")
                st.plotly_chart(fig_cpa, width="stretch", key="efficiency_cpa_chart")

                # Plot 2: Optimal Allocation Stacked Area
                fig_stack = px.area(df_alloc, x="Total Weekly Spend", y="Optimal Allocation ($)", 
                                   color="Channel", title="Optimal Budget Mix at Scale",
                                   color_discrete_sequence=GOOGLE_PALETTE)
                st.plotly_chart(fig_stack, width="stretch", key="efficiency_allocation_chart")
                
                st.success("Analysis complete! The charts show how your most efficient mix changes as budget grows.")
            else:
                st.error("Analysis failed. Ensure the model is trained.")
        else:
            st.info("Click 'Run Efficiency Analysis' to generate the scale-up strategy.")

with tabs[4]:
    st.header("Stage 5: Reach & Frequency Analysis")
    st.markdown("Analyze the reach and frequency dynamics for your R&F enabled channels.")
    
    rf_channels = [c for c in channels if c.get("type") == "Reach & Frequency"]
    
    if not rf_channels:
        st.warning("No Reach & Frequency channels configured in Stage 1.")
    else:
        selected_rf = st.selectbox("Select R&F Channel", [c["name"] for c in rf_channels])
        channel_cfg = next(c for c in rf_channels if c["name"] == selected_rf)
        
        # Modeling parameters from Stage 1
        alpha = channel_cfg.get("saturation", {}).get("params", {}).get("alpha", 1.0)
        gamma = channel_cfg.get("saturation", {}).get("params", {}).get("gamma", 5000.0)
        rf_meta = channel_cfg.get("rf_params", {"max_reach": 0.8, "reach_slope": 1.0})
        
        col_rf1, col_rf2 = st.columns(2)
        
        with col_rf1:
            st.subheader("Audience Reach Curve")
            # Reach = Pop * Max_R * (1 - exp(-slope * spend / 100k))
            spend_range = np.linspace(0, 20000, 100)
            reach_range = 1000000 * rf_meta["max_reach"] * (1 - np.exp(-rf_meta["reach_slope"] * spend_range / 100000))
            
            fig_reach = px.line(x=spend_range, y=reach_range, 
                               title=f"{selected_rf}: Reach vs Weekly Spend",
                               labels={'x': 'Weekly Spend ($)', 'y': 'Unique Audience Reach'},
                               color_discrete_sequence=[G_BLUE])
            st.plotly_chart(fig_reach, width="stretch", key="rf_reach_chart")
            
        with col_rf2:
            st.subheader("Effective Frequency (Hill)")
            # In Meridian, R&F impact is Reach * Hill(Freq)
            freq_range = np.linspace(1, 20, 100)
            # We'll use a standardized Hill for frequency visualization
            # Usually half_sat for freq is small (e.g. 2-5)
            freq_hill = (freq_range**alpha) / (freq_range**alpha + 5**alpha) # Using 5 as half-sat for freq viz
            
            fig_freq = px.line(x=freq_range, y=freq_hill, 
                              title=f"{selected_rf}: Frequency Impact Curve",
                              labels={'x': 'Average Weekly Frequency', 'y': 'Relative Impact per Reached User'},
                              color_discrete_sequence=[G_GREEN])
            st.plotly_chart(fig_freq, width="stretch", key="rf_freq_chart")
            
        st.markdown("---")
        st.subheader("Optimal Frequency Insight")
        st.info(f"Based on the learned parameters, the '{selected_rf}' channel achieves its highest marginal efficiency when users are reached multiple times. The Hill shape (Alpha={alpha:.2f}) indicates how quickly the message 'sticks' vs. when it becomes 'wear-out'.")

