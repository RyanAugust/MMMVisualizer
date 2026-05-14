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
        
        # Treatment for R&F channels in PySiMMMulator: 
        # We'll treat them as Impressions for the spend generation.
        c_impressions = [c["name"] for c in channels if c["type"] in ["Impressions", "Reach & Frequency"]]
        c_clicks = [c["name"] for c in channels if c["type"] == "Clicks"]
        all_channel_names = c_clicks + c_impressions
        
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

        mp = MediaParameters(
            true_cpm={c["name"]: c["true_cost"] for c in channels if c["type"] in ["Impressions", "Reach & Frequency"]},
            true_cpc={c["name"]: c["true_cost"] for c in channels if c["type"] == "Clicks"},
            noisy_cpm_cpc={c["name"]: c.get("cost_noise", {"loc": 0.0, "scale": 0.05}) for c in channels}
        )

        cp = CVRParameters(
            noisy_cvr={c["name"]: c.get("cvr_noise", {"loc": 0.0, "scale": 0.05}) for c in channels}
        )

        adp = AdstockParameters(
            adstock={c["name"]: c.get("adstock", {"type": "geometric", "params": {"lambda_": 0.5}}) for c in channels},
            saturation={c["name"]: c.get("saturation", {"type": "hill", "params": {"alpha": 1.0, "gamma": 5000}}) for c in channels}
        )

        op = OutputParameters(aggregation_level="daily")

        return bp, blp, asp, mp, cp, adp, op

    def generate_data(self) -> pd.DataFrame:
        """Generates synthetic data using the full PySiMMMulator pipeline."""
        if not self.config["channels"]:
            raise ValueError("No channels configured.")

        bp, blp, asp, mp, cp, adp, op = self._prepare_pysimmm_params()
        sim = Simulate(basic_params=bp, random_seed=42)
        
        df_baseline = sim.simulate_baseline(params=blp)
        df = sim.simulate_ad_spend(baseline_sales_df=df_baseline, params=asp)
        df = sim.simulate_media(spend_df=df, params=mp)
        df = sim.simulate_cvr(spend_df=df, params=cp)
        df = sim.simulate_decay_returns(spend_df=df, params=adp)
        df = sim.calculate_conversions(mmm_df=df)
        df = sim.consolidate_dataframe(mmm_df=df, baseline_sales_df=df_baseline)
        
        # Manual R&F generation for R&F channels
        for c in self.config["channels"]:
            if c["type"] == "Reach & Frequency":
                name = c["name"]
                spend = df[f"{name}_spend"]
                # Reach model: logarithmic saturation of population
                # Proxy population = 1,000,000
                pop = 1000000
                rf_cfg = c.get("rf_params", {"max_reach": 0.8, "reach_slope": 1.0})
                max_r = rf_cfg["max_reach"]
                slope_r = rf_cfg["reach_slope"]
                
                # reach = pop * max_r * (1 - exp(-slope * spend / (pop * cost)))
                # we'll simplify:
                df[f"{name}_reach"] = pop * max_r * (1 - np.exp(-slope_r * spend / 100000))
                # frequency = spend / (cost * reach)
                # impressions = reach * frequency
                # so frequency = impressions / reach
                df[f"{name}_frequency"] = df[f"{name}_impressions"] / df[f"{name}_reach"].replace(0, 1)

        revenue_per_conv = self.config["basic"]["revenue_per_conv"]
        df["total_conversions"] = df["total_revenue"] / revenue_per_conv
        
        return df

    def train_model(self):
        """Trains an actual Google Meridian model on generated data."""
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
        freq_cols = [f"{c['name']}_frequency" for c in channels if c["type"] == "Reach & Frequency"]
        rf_spend_cols = [f"{c['name']}_spend" for c in channels if c["type"] == "Reach & Frequency"]

        builder = DataFrameInputDataBuilder(kpi_type=constants.NON_REVENUE)
        builder.with_kpi(df, kpi_col="total_conversions")
        
        if media_channels:
            builder.with_media(df, media_cols=media_cols, media_spend_cols=media_spend_cols, media_channels=media_channels)
        
        if rf_channels:
            builder.with_reach(
                df, 
                reach_cols=reach_cols, 
                frequency_cols=freq_cols, 
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

    def predict(self, spend_decisions: Dict[str, float]) -> Dict[str, Any]:
        """Predicts results using parameters from the trained Meridian model."""
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

            from meridian.analysis.analyzer import Analyzer
            analyzer = Analyzer(mmm)
            rois = analyzer.roi().mean(dim=["chain", "draw"]).to_series()
            
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
                    alpha = post.slope_m.mean(dim=["chain", "draw"]).to_series()[name]
                    gamma = post.ec_m.mean(dim=["chain", "draw"]).to_series()[name]
                    roi_hist = rois[name]
                    
                    sat_hist = (s_hist**alpha) / (s_hist**alpha + gamma**alpha) if s_hist > 0 else 0.5
                    sat_curr = (weekly_spend**alpha) / (weekly_spend**alpha + gamma**alpha) if weekly_spend > 0 else 0
                    beta = (roi_hist * s_hist) / sat_hist if sat_hist > 0 else 0
                    predicted_weekly_rev = beta * sat_curr
                    
                elif name in rf_names:
                    # R&F Impact: Beta * Reach * Hill(Freq)
                    # For prediction we'll assume a constant frequency or similar to historical
                    # Or we could model the spend -> reach shift.
                    # Simplified: treat it like a saturation curve on spend for now in predict
                    # but use the learned R&F parameters if we can.
                    alpha = post.slope_rf.mean(dim=["chain", "draw"]).to_series()[name]
                    gamma = post.ec_rf.mean(dim=["chain", "draw"]).to_series()[name]
                    roi_hist = rois[name]
                    
                    sat_hist = (s_hist**alpha) / (s_hist**alpha + gamma**alpha) if s_hist > 0 else 0.5
                    sat_curr = (weekly_spend**alpha) / (weekly_spend**alpha + gamma**alpha) if weekly_spend > 0 else 0
                    beta = (roi_hist * s_hist) / sat_hist if sat_hist > 0 else 0
                    predicted_weekly_rev = beta * sat_curr
                else:
                    predicted_weekly_rev = 0
                
                predicted_daily_rev = predicted_weekly_rev / 7
                total_predicted_revenue += predicted_daily_rev
                channel_results.append({"channel": name, "spend": daily_spend, "predicted_revenue": predicted_daily_rev})
                
            return {"total_predicted_revenue": float(total_predicted_revenue), "channel_breakdown": channel_results}
        except Exception as e:
            # Simple fallback
            print(f"Prediction error: {e}")
            return {"total_predicted_revenue": 0.0, "channel_breakdown": [], "error": str(e)}

    def optimize_budget(self, total_weekly_budget: float, fixed_allocations: Dict[str, float] = None) -> Dict[str, Any]:
        # Same optimization logic, now unified for both media and rf by treating them as saturation curves
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
                from meridian.analysis.analyzer import Analyzer
                analyzer = Analyzer(mmm)
                rois = analyzer.roi().mean(dim=["chain", "draw"]).to_series()
                
                # Combine media and rf parameters
                slopes = {}
                ecs = {}
                if "media_channel" in post.coords:
                    slopes.update(post.slope_m.mean(dim=["chain", "draw"]).to_series().to_dict())
                    ecs.update(post.ec_m.mean(dim=["chain", "draw"]).to_series().to_dict())
                if "rf_channel" in post.coords:
                    slopes.update(post.slope_rf.mean(dim=["chain", "draw"]).to_series().to_dict())
                    ecs.update(post.ec_rf.mean(dim=["chain", "draw"]).to_series().to_dict())

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
