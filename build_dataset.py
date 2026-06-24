# -*- coding: utf-8 -*-
"""
build_dataset.py
----------------
Stage 1 of the maritime-construction safety pipeline.

Loads the OSHA records, applies the (leakage-free) maritime filter, reconstructs
micro-climate weather for each incident via the Meteostat API, runs the rule-based
NLP ontology, and engineers the modelling features.

The enriched table is CACHED to 'enriched_maritime_data.parquet' so that figures
and tables are fully REPRODUCIBLE and do not require re-querying the weather API on
every run. Delete the parquet to force a fresh rebuild.
"""
import os
import re
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Repo-relative paths: place the OSHA CSV in ../data next to this src/ folder
# (or set the OSHA_CSV environment variable to point elsewhere).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")
FILE_PATH = os.environ.get("OSHA_CSV", os.path.join(_DATA, "January2015toFebruary2025.csv"))
CACHE_PATH = os.path.join(_DATA, "enriched_maritime_data.parquet")

MARITIME_NAICS = ["237990", "237110", "238990", "488390", "336611", "237310"]
MARITIME_KEYWORDS = [
    "marine", "maritime", "port", "harbor", "harbour", "dock", "pier",
    "wharf", "vessel", "ship", "boat", "offshore", "underwater", "diving",
    "dredge", "dredging", "barge", "tugboat", "anchor", "mooring",
    "jetty", "breakwater", "seawall", "shipyard", "drydock", "platform",
    "oil rig", "drilling platform", "subsea", "coastal",
]
COASTAL_STATES = [
    "ALASKA", "WASHINGTON", "OREGON", "CALIFORNIA", "TEXAS", "LOUISIANA",
    "MISSISSIPPI", "ALABAMA", "FLORIDA", "GEORGIA", "SOUTH CAROLINA",
    "NORTH CAROLINA", "VIRGINIA", "MARYLAND", "DELAWARE", "NEW JERSEY",
    "NEW YORK", "CONNECTICUT", "RHODE ISLAND", "MASSACHUSETTS",
    "NEW HAMPSHIRE", "MAINE", "HAWAII", "PUERTO RICO",
]


# ----------------------------------------------------------------------------
# SECTION 1: DATA LOADING & LEAKAGE-FREE EMPLOYER HISTORY
# ----------------------------------------------------------------------------
def load_and_filter_data(filepath):
    print("[Step 1] Loading and filtering maritime data ...")
    df = pd.read_csv(filepath, low_memory=False)
    df["Primary NAICS"] = df["Primary NAICS"].astype(str)

    cdf = df[df["Primary NAICS"].isin(MARITIME_NAICS)].copy()
    cdf["search_text"] = (
        cdf["Final Narrative"].fillna("") + " "
        + cdf["Employer"].fillna("") + " "
        + cdf["Address1"].fillna("") + " "
        + cdf["Address2"].fillna("") + " "
        + cdf["City"].fillna("")
    ).str.lower()
    mask = cdf["search_text"].apply(lambda x: any(k in x for k in MARITIME_KEYWORDS))
    mdf = cdf[mask].copy()
    mdf = mdf[mdf["State"].isin(COASTAL_STATES)].copy()

    mdf["EventDate"] = pd.to_datetime(mdf["EventDate"], errors="coerce")
    mdf = mdf.dropna(subset=["Latitude", "Longitude", "EventDate"])
    mdf["Hospitalized"] = mdf["Hospitalized"].fillna(0).astype(int)
    if "Amputation" not in mdf.columns:
        mdf["Amputation"] = 0
    else:
        mdf["Amputation"] = pd.to_numeric(mdf["Amputation"], errors="coerce").fillna(0).astype(int)

    # ---- Leakage-free employer history (uses ONLY prior incidents) ----------
    mdf = mdf.sort_values(["Employer", "EventDate"])
    mdf["past_incidents"] = mdf.groupby("Employer").cumcount()
    mdf["past_severe"] = (
        mdf.groupby("Employer")["Hospitalized"]
        .transform(lambda x: x.shift(1).cumsum())
        .fillna(0)
    )
    mdf["employer_historical_severity"] = mdf["past_severe"] / (mdf["past_incidents"] + 1)
    mdf["employer_is_high_severity"] = (mdf["employer_historical_severity"] > 0.5).astype(int)

    mdf = mdf.sort_values("EventDate").reset_index(drop=True)
    print(f"  Original records:           {len(df)}")
    print(f"  Maritime incidents (final): {len(mdf)}")
    print(f"  Hospitalization rate:       {(mdf['Hospitalized'] > 0).mean():.3f}")
    return mdf


# ----------------------------------------------------------------------------
# SECTION 2: WEATHER RECONSTRUCTION (Meteostat) WITH CACHING
# ----------------------------------------------------------------------------
def _get_weather_single(args):
    from meteostat import Stations, Hourly, Daily
    lat, lon, date, idx = args
    try:
        lat, lon = float(lat), float(lon)
        start = datetime(date.year, date.month, date.day)
        end = start + timedelta(days=1)
        station = Stations().nearby(lat, lon).fetch(1)
        if station.empty:
            return idx, None
        sid = station.index[0]
        hourly = Hourly(sid, start, end).fetch()
        if hourly.empty:
            daily = Daily(sid, start, end).fetch()
            if daily.empty:
                return idx, None
            r = daily.iloc[0]
            wd = {
                "temp_mean": float(r.get("tavg", np.nan)),
                "temp_max": float(r.get("tmax", np.nan)),
                "temp_min": float(r.get("tmin", np.nan)),
                "temp_variance": 0.0,
                "precip_total": float(r.get("prcp", 0.0)),
                "wind_speed_mean": float(r.get("wspd", 0.0)),
            }
        else:
            wd = {
                "temp_mean": float(hourly["temp"].mean()),
                "temp_max": float(hourly["temp"].max()),
                "temp_min": float(hourly["temp"].min()),
                "temp_variance": float(hourly["temp"].var()),
                "precip_total": float(hourly["prcp"].sum()),
                "wind_speed_mean": float(hourly["wspd"].mean()),
            }
        if pd.isna(wd["temp_mean"]):
            return idx, None
        return idx, wd
    except Exception:
        return idx, None


def batch_weather(df, max_workers=50):
    import concurrent.futures
    print(f"[Step 2] Fetching weather (parallel workers={max_workers}) ...")
    args_list = [(r["Latitude"], r["Longitude"], r["EventDate"], i) for i, r in df.iterrows()]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_get_weather_single, a) for a in args_list]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            idx, w = fut.result()
            results[idx] = w
            done += 1
            if done % 200 == 0:
                print(f"  weather progress: {done}/{len(args_list)}")

    valid_idx = [i for i in df.index if results.get(i) is not None]
    wdf = pd.DataFrame([results[i] for i in valid_idx], index=valid_idx)
    out = pd.concat(
        [df.loc[valid_idx].reset_index(drop=True), wdf.reset_index(drop=True)], axis=1
    )
    for c in ["temp_mean", "temp_max", "temp_min", "temp_variance", "precip_total", "wind_speed_mean"]:
        out[c] = out[c].fillna(out[c].median())

    # Derived environmental indicators
    out["month"] = out["EventDate"].dt.month
    out["extreme_heat"] = (out["temp_max"] > 35).astype(int)
    out["freeze_thaw"] = ((out["temp_min"] < 0) & (out["temp_max"] > 0)).astype(int)
    out["high_wind"] = (out["wind_speed_mean"] > 15).astype(int)
    print(f"  Incidents with valid weather: {len(out)}")
    return out


# ----------------------------------------------------------------------------
# SECTION 3: RULE-BASED NLP ONTOLOGY (15 equipment classes)
# ----------------------------------------------------------------------------
EQUIPMENT_PATTERNS = {
    "Floating Crane":     r"\b(crane|hoist|derrick|gantry|winch)\w*",
    "Vessel/Barge":       r"\b(vessel|ship|boat|barge|tug|skiff)\w*",
    "Scaffold/Platform":  r"\b(scaffold|platform|staging|plank)\w*",
    "Ladder/Gangway":     r"\b(ladder|gangway|ramp)\w*",
    "Forklift/Lift":      r"\b(forklift|lift|reach\s*truck)\w*",
    "Excavator/Dredge":   r"\b(excavat|dredge|backhoe|loader)\w*",
    "Welding Apparatus":  r"\b(weld|torch|burn|cutting|slag)\w*",
    "Electrical":         r"\b(panel|breaker|wire|electric|circuit|voltage)\w*",
    "Pump/Compressor":    r"\b(pump|compressor|pneumatic|hose)\w*",
    "Rigging/Sling":      r"\b(rigging|sling|shackle|cable|wire\s*rope|chain)\w*",
    "Saw/Grinder":        r"\b(saw|grind|blade|cut\s*off)\w*",
    "Hand Tool":          r"\b(hammer|wrench|drill|chisel|hand\s*tool)\w*",
    "Vehicle/Truck":      r"\b(truck|vehicle|van|trailer|forklift)\w*",
    "Pile Driver":        r"\b(pile|piling|sheet\s*pile)\w*",
    "Conveyor":           r"\b(conveyor|belt)\w*",
}
MECHANICAL_PAT = r"\b(broke|fail|malfunction|rupture|collapse|corrode|rust|defect|crack)\w*"
ENV_PAT = r"\b(wave|tide|current|wind|storm|weather|sea\s*state|swell|rain|snow|ice|heat|cold|fog)\w*"
OPERATOR_PAT = r"\b(slip|fell|fall|struck|caught|pinned|drop|crush|misstep|trip)\w*"


def extract_nlp(narratives):
    print("[Step 3] Extracting NLP equipment / error-type features ...")
    rows = []
    for nar in narratives:
        if pd.isna(nar):
            rows.append({"nlp_equipment": "Other/Unknown", "error_type": "ambiguous", "environmental_mention": 0})
            continue
        s = str(nar).lower()
        equip = "Other/Unknown"
        for name, pat in EQUIPMENT_PATTERNS.items():
            if re.search(pat, s):
                equip = name
                break
        mech = len(re.findall(MECHANICAL_PAT, s))
        oper = len(re.findall(OPERATOR_PAT, s))
        env = len(re.findall(ENV_PAT, s))
        if mech > oper:
            et = "mechanical"
        elif oper > mech:
            et = "operator"
        else:
            et = "ambiguous"
        rows.append({"nlp_equipment": equip, "error_type": et, "environmental_mention": int(env > 0)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def build(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        print(f"Cache found -> {CACHE_PATH} (delete to rebuild)")
        return pd.read_parquet(CACHE_PATH)

    df = load_and_filter_data(FILE_PATH)
    df = batch_weather(df, max_workers=50)
    nlp = extract_nlp(df["Final Narrative"])
    df = pd.concat([df.reset_index(drop=True), nlp.reset_index(drop=True)], axis=1)

    # drop non-serializable / unused text helper
    df = df.drop(columns=["search_text"], errors="ignore")
    df.to_parquet(CACHE_PATH, index=False)
    print(f"\nSaved enriched dataset -> {CACHE_PATH}  (n={len(df)})")
    return df


if __name__ == "__main__":
    out = build(force=False)
    print("\nEquipment distribution:")
    print(out["nlp_equipment"].value_counts())
    print("\nColumns:", len(out.columns))
