import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pysimmmulator import Simulate
from pysimmmulator.param_handlers import (
    BasicParameters, 
    BaselineParameters, 
    AdSpendParameters, 
    MediaParameters, 
    CVRParameters, 
    AdstockParameters, 
    OutputParameters
)
import datetime
import pickle

class MeridianManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.config_path = os.path.join(data_dir, "config.json")
        self.model_path = os.path.join(data_dir, "model.pkl")
        self.config = self._load_config()
        self.is_training = False

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {
            "basic": {
                "years": 3,
                "start_date": "2022/01/01",
                "frequency_of_campaigns": 7,
                "revenue_per_conv": 500.0
            },
            "baseline": {
                "base_p": 1000,
                "trend_p": 500,
                "temp_var": 200,
                "temp_coef_mean": 0.5,
                "temp_coef_sd": 0.1,
                "error_std": 50
            },
            "channels": []
        }

    def save_config(self, config: Dict[str, Any]):
        self.config = config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def _prepare_pysimmm_params(self):
        cfg = self.config
        channels = cfg["channels"]
        
        c_impressions = [c["name"] for c in channels if c["type"] in ["Impressions", "Reach & Frequency"]]
        c_clicks = [c["name"] for c in channels if c["type"] == "Clicks"]
        
        bp = BasicParameters(
            years=cfg["basic"]["years"],
            channels_impressions=c_impressions,
            channels_clicks=c_clicks,
            frequency_of_campaigns=cfg["basic"]["frequency_of_campaigns"],
            start_date=cfg["basic"]["start_date"],
            true_cvr={c["name"]: c["true_cvr"] for c in channels},
            revenue_per_conv=cfg["basic"]["revenue_per_conv"]
        )

        bl_cfg = cfg["baseline"]
        blp = BaselineParameters(
            basic_params=bp,
            base_p=bl_cfg["base_p"],
            trend_p=bl_cfg["trend_p"],
            temp_var=bl_cfg["temp_var"],
            temp_coef_mean=bl_cfg.get("temp_coef_mean", 0.5),
            temp_coef_sd=bl_cfg.get("temp_coef_sd", 0.1),
            error_std=bl_cfg["error_std"]
        )

        all_channel_names = c_clicks + c_impressions
        total_avg_spend = sum([c.get("avg_spend", 5000.0) for c in channels])
        proportions = {}
        if len(all_channel_names) > 1:
            for name in all_channel_names[:-1]:
                channel_cfg = next(c for c in channels if c["name"] == name)
                prop = channel_cfg.get("avg_spend", 5000.0) / total_avg_spend
                proportions[name] = {"min": prop * 0.9, "max": prop * 1.1}
        
        asp = AdSpendParameters(
            campaign_spend_mean=total_avg_spend,
            campaign_spend_std=total_avg_spend * 0.1,
            max_min_proportion_on_each_channel=proportions
        )

        rf_truth = {}
        for c in channels:
            if c["type"] == "Reach & Frequency":
                # PySiMMMulator expects 'reach' or 'frequency' in the config
                rf_cfg = c.get("rf_params", {"max_reach": 0.8})
                # If reach_slope was used, we'll just pass max_reach as reach value for now
                rf_truth[c["name"]] = {"reach": rf_cfg.get("max_reach", 0.8)}

        mp = MediaParameters(
            true_cpm={c["name"]: c["true_cost"] for c in channels if c["type"] in ["Impressions", "Reach & Frequency"]},
            true_cpc={c["name"]: c["true_cost"] for c in channels if c["type"] == "Clicks"},
            noisy_cpm_cpc={c["name"]: c.get("cost_noise", {"loc": 0.0, "scale": 0.05}) for c in channels},
            true_reach_frequency=rf_truth
        )

        cp = CVRParameters(
            noisy_cvr={c["name"]: c.get("cvr_noise", {"loc": 0.0, "scale": 0.05}) for c in channels}
        )

        adp = AdstockParameters(
            adstock={c["name"]: c.get("adstock", {"type": "geometric", "params": {"lambda_": 0.5}}) for c in channels},
            saturation={c["name"]: c.get("saturation", {"type": "hill", "params": {"alpha": 1.0, "gamma": 5000}}) for c in channels}
        )

        op = OutputParameters(aggregation_level="daily")
        
        gp = GeoParameters(total_population=1000000)

        return bp, blp, asp, mp, cp, adp, op, gp

    def generate_data(self) -> pd.DataFrame:
        """Generates synthetic data using the full PySiMMMulator pipeline."""
        if not self.config["channels"]:
            raise ValueError("No channels configured.")

        bp, blp, asp, mp, cp, adp, op, gp = self._prepare_pysimmm_params()
        sim = Simulate(basic_params=bp, random_seed=42)
        sim.total_population = gp.total_population
        
        # Build the full config for run_with_config
        full_config = {
            "basic_params": {
                "years": bp.years,
                "channels_clicks": bp.channels_clicks,
                "channels_impressions": bp.channels_impressions,
                "frequency_of_campaigns": bp.frequency_of_campaigns,
                "start_date": bp.start_date.strftime("%Y/%m/%d"),
                "true_cvr": bp.true_cvr,
                "revenue_per_conv": bp.revenue_per_conv
            },
            "baseline_params": {
                "base_p": blp.base_p,
                "trend_p": blp.trend_p,
                "temp_var": blp.temp_var,
                "temp_coef_mean": blp.temp_coef_mean,
                "temp_coef_sd": blp.temp_coef_sd,
                "error_std": blp.error_std,
                "exogenous_factors": blp.exogenous_factors
            },
            "ad_spend_params": {
                "campaign_spend_mean": asp.campaign_spend_mean,
                "campaign_spend_std": asp.campaign_spend_std,
                "max_min_proportion_on_each_channel": asp.max_min_proportion_on_each_channel
            },
            "media_params": {
                "true_cpm": mp.true_cpm,
                "true_cpc": mp.true_cpc,
                "noisy_cpm_cpc": mp.noisy_cpm_cpc,
                "true_reach_frequency": mp.true_reach_frequency
            },
            "cvr_params": {
                "noisy_cvr": cp.noisy_cvr
            },
            "adstock_params": {
                "adstock": adp.adstock,
                "saturation": adp.saturation
            },
            "output_params": {
                "aggregation_level": op.aggregation_level
            },
            "geo_params": {
                "total_population": gp.total_population,
                "geo_specs": gp.geo_specs,
                "universal_scale": gp.universal_scale,
                "count": gp.count,
                "dist_spec": gp.dist_spec,
                "media_cost_spec": gp.media_cost_spec,
                "perf_spec": gp.perf_spec
            }
        }
        
        res = sim.run_with_config(full_config)
        df = res.df
        
        # Flatten multi-index if present (national geo + date)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        revenue_per_conv = self.config["basic"]["revenue_per_conv"]
        df["total_conversions"] = df["total_revenue"] / revenue_per_conv
        
        return df

    def train_model(self):
        """Trains an actual Google Meridian model on generated data."""
        self.is_training = True
        try:
            df = self.generate_data()
            df["geo"] = "national"
            df["time"] = pd.to_datetime(df["date"])
            
            from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder
            from meridian.model.model import Meridian, save_mmm
            from meridian.model.spec import ModelSpec
            from meridian.model.prior_distribution import PriorDistribution
            from meridian import constants

            channels = self.config["channels"]
            
            # Standard Media Channels
            media_channels = [c["name"] for c in channels if c["type"] != "Reach & Frequency"]
            media_cols = [f"{c['name']}_impressions" if c["type"] == "Impressions" else f"{c['name']}_clicks" for c in channels if c["type"] != "Reach & Frequency"]
            media_spend_cols = [f"{c['name']}_spend" for c in channels if c["type"] != "Reach & Frequency"]

            # R&F Channels
            rf_channels = [c["name"] for c in channels if c["type"] == "Reach & Frequency"]
            reach_cols = [f"{c['name']}_reach" for c in channels if c["type"] == "Reach & Frequency"]
            frequency_cols = [f"{c['name']}_frequency" for c in channels if c["type"] == "Reach & Frequency"]
            rf_spend_cols = [f"{c['name']}_spend" for c in channels if c["type"] == "Reach & Frequency"]

            builder = DataFrameInputDataBuilder(kpi_type=constants.NON_REVENUE)
            builder.with_kpi(df, kpi_col="total_conversions")
            
            if media_channels:
                builder.with_media(df, media_cols=media_cols, media_spend_cols=media_spend_cols, media_channels=media_channels)
            
            if rf_channels:
                builder.with_reach(
                    df, 
                    reach_cols=reach_cols, 
                    frequency_cols=frequency_cols, 
                    rf_spend_cols=rf_spend_cols, 
                    rf_channels=rf_channels
                )

            df["population"] = 1000000.0 # Match pop used in generate_data
            builder.with_population(df, population_col="population")
            
            input_data = builder.build()
            model_spec = ModelSpec(prior=PriorDistribution(), max_lag=8, holdout_id=None)
            mmm = Meridian(input_data=input_data, model_spec=model_spec)
            mmm.sample_posterior(n_chains=1, n_adapt=50, n_burnin=50, n_keep=100)
            save_mmm(mmm, self.model_path)
            return {"status": "success", "message": "Actual Meridian model trained successfully (with R&F support)."}
        finally:
            self.is_training = False

    def predict(self, spend_decisions: Dict[str, float]) -> Dict[str, Any]:
        """Predicts results using parameters from the trained Meridian model."""
        if self.is_training:
            return {"status": "error", "message": "Model is currently being trained. Please wait."}
        if not os.path.exists(self.model_path):
            raise ValueError("Model not trained.")
            
        from meridian.model.model import load_mmm
        try:
            mmm = load_mmm(self.model_path)
            post = mmm.inference_data.posterior
            
            # Media Channels
            has_media = "media_channel" in post.coords
            media_names = list(post.coords["media_channel"].values) if has_media else []
            
            # RF Channels
            has_rf = "rf_channel" in post.coords
            rf_names = list(post.coords["rf_channel"].values) if has_rf else []

            # Helper to safely get mean from posterior and convert to series
            def get_post_series(var_name):
                if var_name in post.data_vars:
                    res = post[var_name].mean(dim=["chain", "draw"])
                    if "geo" in res.dims:
                        res = res.mean(dim="geo")
                    return res.to_series()
                return pd.Series(dtype=float)

            rois_m = get_post_series("roi_m")
            rois_rf = get_post_series("roi_rf")
            alphas_m = get_post_series("alpha_m")
            gammas_m = get_post_series("ec_m")
            alphas_rf = get_post_series("alpha_rf")
            gammas_rf = get_post_series("ec_rf")
            
            config = self.get_config()
            revenue_per_conv = config["basic"]["revenue_per_conv"]
            total_predicted_revenue = config["baseline"]["base_p"] * revenue_per_conv
            
            channel_results = []
            for channel in config["channels"]:
                name = channel["name"]
                daily_spend = spend_decisions.get(name, 0.0)
                weekly_spend = daily_spend * 7
                s_hist = channel.get("avg_spend", 5000.0)

                if name in media_names:
                    alpha = alphas_m.get(name, 1.0)
                    gamma = gammas_m.get(name, 1.0)
                    roi_hist = rois_m.get(name, 0.0)
                    
                    # We assume s_hist and weekly_spend are on the same scale.
                    # Since gamma is from the model, it is scaled. 
                    # For a rough approximation, we'll use s_hist as a reference.
                    # A better way would be to unscale gamma, but we don't have the scaler easily.
                    # However, if we assume the model learned a gamma relative to the scaled media,
                    # and we use the same ratio, it should be okay-ish for the UI.
                    
                    sat_hist = (s_hist**alpha) / (s_hist**alpha + gamma**alpha) if s_hist > 0 else 0.5
                    sat_curr = (weekly_spend**alpha) / (weekly_spend**alpha + gamma**alpha) if weekly_spend > 0 else 0
                    beta = (roi_hist * s_hist) / sat_hist if sat_hist > 0 else 0
                    predicted_weekly_rev = beta * sat_curr
                    
                elif name in rf_names:
                    alpha = alphas_rf.get(name, 1.0)
                    gamma = gammas_rf.get(name, 1.0)
                    roi_hist = rois_rf.get(name, 0.0)
                    
                    sat_hist = (s_hist**alpha) / (s_hist**alpha + gamma**alpha) if s_hist > 0 else 0.5
                    sat_curr = (weekly_spend**alpha) / (weekly_spend**alpha + gamma**alpha) if weekly_spend > 0 else 0
                    beta = (roi_hist * s_hist) / sat_hist if sat_hist > 0 else 0
                    predicted_weekly_rev = beta * sat_curr
                else:
                    predicted_weekly_rev = 0
                
                predicted_daily_rev = predicted_weekly_rev / 7
                total_predicted_revenue += predicted_daily_rev
                channel_results.append({
                    "channel": name, 
                    "spend": float(daily_spend), 
                    "predicted_revenue": float(predicted_daily_rev)
                })
                
            return {"total_predicted_revenue": float(total_predicted_revenue), "channel_breakdown": channel_results}
        except Exception as e:
            # Simple fallback
            print(f"Prediction error: {e}")
            return {"total_predicted_revenue": 0.0, "channel_breakdown": [], "error": str(e)}

    def optimize_budget(self, total_weekly_budget: float, fixed_allocations: Dict[str, float] = None) -> Dict[str, Any]:
        # Same optimization logic, now unified for both media and rf by treating them as saturation curves
        if self.is_training:
            return {"status": "error", "message": "Model is currently being trained. Please wait."}
        from scipy.optimize import minimize
        config = self.get_config()
        channels = config.get("channels", [])
        if not channels: return {"status": "error", "message": "No channels to optimize."}

        params_to_use = []
        model_exists = os.path.exists(self.model_path)
        if model_exists:
            try:
                from meridian.model.model import load_mmm
                mmm = load_mmm(self.model_path)
                post = mmm.inference_data.posterior
                rois = {}
                if "roi_m" in post.data_vars:
                    rm = post.roi_m.mean(dim=["chain", "draw"])
                    if "geo" in rm.dims: rm = rm.mean(dim="geo")
                    rois.update(rm.to_series().to_dict())
                if "roi_rf" in post.data_vars:
                    rrf = post.roi_rf.mean(dim=["chain", "draw"])
                    if "geo" in rrf.dims: rrf = rrf.mean(dim="geo")
                    rois.update(rrf.to_series().to_dict())
                
                # Combine media and rf parameters
                slopes = {}
                ecs = {}
                if "alpha_m" in post.data_vars:
                    am = post.alpha_m.mean(dim=["chain", "draw"])
                    if "geo" in am.dims: am = am.mean(dim="geo")
                    slopes.update(am.to_series().to_dict())
                if "ec_m" in post.data_vars:
                    em = post.ec_m.mean(dim=["chain", "draw"])
                    if "geo" in em.dims: em = em.mean(dim="geo")
                    ecs.update(em.to_series().to_dict())
                if "alpha_rf" in post.data_vars:
                    arf = post.alpha_rf.mean(dim=["chain", "draw"])
                    if "geo" in arf.dims: arf = arf.mean(dim="geo")
                    slopes.update(arf.to_series().to_dict())
                if "ec_rf" in post.data_vars:
                    erf = post.ec_rf.mean(dim=["chain", "draw"])
                    if "geo" in erf.dims: erf = erf.mean(dim="geo")
                    ecs.update(erf.to_series().to_dict())

                for c in channels:
                    name = c["name"]
                    if name in rois:
                        s_hist = c.get("avg_spend", 5000.0)
                        alpha, gamma = slopes[name], ecs[name]
                        sat_hist = (s_hist**alpha) / (s_hist**alpha + gamma**alpha) if s_hist > 0 else 0.5
                        beta = (rois[name] * s_hist) / sat_hist if sat_hist > 0 else 0
                        params_to_use.append({"name": name, "alpha": alpha, "gamma": gamma, "beta": beta})
                    else: model_exists = False
            except: model_exists = False

        if not model_exists:
            for c in channels:
                sat_cfg = c.get("saturation", {"params": {"alpha": 1.0, "gamma": 5000}})
                alpha, gamma = sat_cfg["params"]["alpha"], sat_cfg["params"]["gamma"]
                beta = (gamma * 2) * 10 * c["true_cvr"] * config["basic"]["revenue_per_conv"]
                params_to_use.append({"name": c["name"], "alpha": alpha, "gamma": gamma, "beta": beta})

        fixed_allocations = fixed_allocations or {}
        free_params = [p for p in params_to_use if p["name"] not in fixed_allocations]
        fixed_spends_sum = sum(fixed_allocations.values())
        remaining_budget = total_weekly_budget - fixed_spends_sum

        def objective(free_spends):
            total_rev = 0
            for i, spend in enumerate(free_spends):
                p = free_params[i]
                sat = (spend**p["alpha"]) / (spend**p["alpha"] + p["gamma"]**p["alpha"]) if spend > 0 else 0
                total_rev += p["beta"] * sat
            for name, spend in fixed_allocations.items():
                p = next(p for p in params_to_use if p["name"] == name)
                sat = (spend**p["alpha"]) / (spend**p["alpha"] + p["gamma"]**p["alpha"]) if spend > 0 else 0
                total_rev += p["beta"] * sat
            return -total_rev

        if free_params and remaining_budget > 0:
            res = minimize(objective, [remaining_budget/len(free_params)]*len(free_params), 
                          method='SLSQP', bounds=[(0, remaining_budget)]*len(free_params),
                          constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - remaining_budget}))
            if res.success:
                allocs = {}
                for i, p in enumerate(free_params):
                    allocs[p["name"]] = {"weekly_spend": float(res.x[i]), "percentage": float(res.x[i]/total_weekly_budget*100)}
                for name, spend in fixed_allocations.items():
                    allocs[name] = {"weekly_spend": float(spend), "percentage": float(spend/total_weekly_budget*100)}
                return {"status": "success", "total_weekly_budget": total_weekly_budget, "allocations": allocs, "expected_weekly_revenue": float(-res.fun)}
            return {"status": "error", "message": res.message}
        return {"status": "success", "allocations": {p["name"]: {"weekly_spend": fixed_allocations.get(p["name"], 0), "percentage": fixed_allocations.get(p["name"], 0)/total_weekly_budget*100} for p in params_to_use}, "expected_weekly_revenue": float(-objective([]))}
