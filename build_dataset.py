# -*- coding: utf-8 -*-
"""
build_dataset.py
----------------
Stage 1 of the maritime-construction safety pipeline.

Loads the OSHA records, applies the maritime filter, reconstructs
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
FILE_PATH = os.environ.get("OSHA_CSV", os.path.join(_DATA, "January2015toAugust2025.csv"))
CACHE_PATH = os.path.join(_DATA, "enriched_maritime_data.parquet")

MARITIME_NAICS = ["237990", "237110", "238990", "488390", "336611", "237310"]
MARITIME_KEYWORDS = [
    "marine", "maritime", "port", "harbor", "harbour", "dock", "pier",
    "wharf", "vessel", "ship", "boat", "offshore", "underwater", "diving",
    "dredge", "dredging", "barge", "tugboat", "anchor", "mooring",
    "jetty", "breakwater", "seawall", "shipyard", "drydock", "platform",
    "oil rig", "drilling platform", "subsea", "coastal",
]
BROAD_MARITIME_KEYWORDS = MARITIME_KEYWORDS + [
    "bridge", "river", "canal", "lock", "dam", "levee", "bulkhead",
    "shore", "shoreline", "waterfront", "marina", "naval", "navy",
    "coast guard", "ferry", "submarine", "tanker", "launch", "dockyard",
    "quay", "slipway", "boatyard", "gulf", "bay", "ocean", "sea",
    "lake", "reservoir", "channel",
]
BBOX_LAT = (24, 50)
BBOX_LON = (-125, -65)
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
    mask = cdf["search_text"].apply(lambda x: any(k in x for k in BROAD_MARITIME_KEYWORDS))
    mdf = cdf[mask].copy()

    mdf["EventDate"] = pd.to_datetime(mdf["EventDate"], errors="coerce")
    mdf = mdf.dropna(subset=["Latitude", "Longitude", "EventDate"])
    mdf["Latitude"] = pd.to_numeric(mdf["Latitude"], errors="coerce")
    mdf["Longitude"] = pd.to_numeric(mdf["Longitude"], errors="coerce")
    mdf = mdf[
        mdf["Latitude"].between(*BBOX_LAT)
        & mdf["Longitude"].between(*BBOX_LON)
        & mdf["Final Narrative"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    mdf["Hospitalized"] = mdf["Hospitalized"].fillna(0).astype(int)
    if "Amputation" not in mdf.columns:
        mdf["Amputation"] = 0
    else:
        mdf["Amputation"] = pd.to_numeric(mdf["Amputation"], errors="coerce").fillna(0).astype(int)

    mdf = mdf.sort_values("EventDate").reset_index(drop=True)
    print(f"  Original records:           {len(df)}")
    print(f"  Broad maritime-bbox records before NLP drop: {len(mdf)}")
    print(f"  Hospitalization rate:       {(mdf['Hospitalized'] > 0).mean():.3f}")
    return mdf


def add_employer_history(df):
    """Leakage-free employer history computed only from prior cohort records."""
    out = df.sort_values(["Employer", "EventDate"]).copy()
    out["past_incidents"] = out.groupby("Employer").cumcount()
    out["past_severe"] = (
        out.groupby("Employer")["Hospitalized"]
        .transform(lambda x: x.shift(1).cumsum())
        .fillna(0)
    )
    out["employer_historical_severity"] = out["past_severe"] / (out["past_incidents"] + 1)
    out["employer_is_high_severity"] = (out["employer_historical_severity"] > 0.5).astype(int)
    return out.sort_values("EventDate").reset_index(drop=True)


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
    "Vessel/Barge":       r"\b(vessel|ship|boat|barge|tug|skiff|submarine|tanker|aircraft\s*carrier)\w*",
    "Scaffold/Platform":  r"\b(scaffold|platform|staging|plank|catwalk|walkway)\w*",
    "Ladder/Gangway":     r"\b(ladder|gangway|ramp|stair|stairway|step)\w*",
    "Forklift/Lift":      r"\b(forklift|fork\s*lift|reach\s*truck|aerial\s*lift|scissor\s*lift|man\s*lift|manlift|boom\s*lift|telehandler)\w*",
    "Excavator/Dredge":   r"\b(excavat|dredge|backhoe|loader|bulldozer|dozer|auger)\w*",
    "Welding Apparatus":  r"\b(weld|torch|cutting\s*torch|hot\s*work|slag|plasma\s*cutter|brazing)\w*",
    "Electrical":         r"\b(panel|breaker|wire|electric|circuit|voltage|energized|power\s*line)\w*",
    "Pump/Compressor":    r"\b(pump|compressor|pneumatic|hose|pressure\s*washer|high[-\s]*pressure\s*wand|blast\s*pot|blasting\s*pot|abrasive\s*blasting)\w*",
    "Rigging/Sling":      r"\b(rigging|sling|shackle|cable|wire\s*rope|chain)\w*",
    "Saw/Grinder":        r"\b(saw|grind|blade|cut\s*off|cutoff|planer)\w*",
    "Hand Tool":          r"\b(hammer|sledgehammer|wrench|drill|chisel|hand\s*tool|pry\s*bar|pocket\s*knife|knife|punch)\w*",
    "Vehicle/Truck":      r"\b(truck|vehicle|van|trailer|transporter|tractor\s*trailer|pickup)\w*",
    "Pile Driver":        r"\b(pile|piling|sheet\s*pile)\w*",
    "Conveyor":           r"\b(conveyor|belt)\w*",
}
SOURCE_FALLBACK_PATTERNS = {
    "Vessel/Barge":       r"\b(water vehicle|ship|vessel|barge|boat|tanker|submarine|ferry)\b",
    "Scaffold/Platform":  r"\b(floor|floors|walkway|walkways|ground surface|constructed surface|floor opening|structural element|structural metal|beams?|rails?|plates?|panels?|plating|grating|deck|catwalk|platform|scaffold|caps?|lids?|covers?|doors?|gates?)\b",
    "Ladder/Gangway":     r"\b(ladder|stairs?|steps?|stairway)\b",
    "Forklift/Lift":      r"\b(forklift|aerial lift|scissor lift|manlift|boom lift|material and personnel handling machinery|pallet jack)\b",
    "Excavator/Dredge":   r"\b(excavation|excavations|trenches|ditches|loader|auger|drilling and extraction machinery)\b",
    "Welding Apparatus":  r"\b(welding|torch|hot work|welding, cutting, and blow torches)\b",
    "Electrical":         r"\b(electric|electrical|power lines?|transformers?|voltage|shock|circuit|wire)\b",
    "Pump/Compressor":    r"\b(pump|compressor|pneumatic|hose|pressurized water|water-blast|power washer|sandblaster|blast|fan|blower|pressure)\b",
    "Rigging/Sling":      r"\b(rigging|sling|shackle|cable|wire rope|chain|lifeline|lanyard|harness|jacks?)\b",
    "Saw/Grinder":        r"\b(saw|grinder|blade|lathe|lathes|special process machinery|metalworking|cutting machinery|planer)\b",
    "Hand Tool":          r"\b(hammer|sledgehammer|wrench|drill|chisel|tool|pry bar|knife|punch|wheelbarrow)\b",
    "Vehicle/Truck":      r"\b(truck|vehicle|van|trailer|automobile|atv|tractor trailer|transporter|semi)\b",
    "Pile Driver":        r"\b(pile|piling|pile driver)\b",
    "Conveyor":           r"\b(conveyor|belt)\b",
}
MECHANICAL_PAT = r"\b(broke|fail|malfunction|rupture|collapse|corrode|rust|defect|crack)\w*"
ENV_PAT = r"\b(wave|tide|current|wind|storm|weather|sea\s*state|swell|rain|snow|ice|heat|cold|fog)\w*"
OPERATOR_PAT = r"\b(slip|fell|fall|struck|caught|pinned|drop|crush|misstep|trip)\w*"


def _classify_equipment(text, patterns):
    s = str(text).lower()
    for name, pat in patterns.items():
        if re.search(pat, s):
            return name
    return "Other/Unknown"


def extract_nlp(records):
    print("[Step 3] Extracting NLP equipment / error-type features ...")
    rows = []
    if isinstance(records, pd.DataFrame):
        iterator = records.iterrows()
    else:
        iterator = enumerate(pd.Series(records, name="Final Narrative").to_frame().iterrows())

    for _, rec in iterator:
        if isinstance(rec, tuple):
            rec = rec[1]
        nar = rec.get("Final Narrative", np.nan)
        source_text = " ".join(
            str(rec.get(c, "") or "")
            for c in ["SourceTitle", "Secondary Source Title", "EventTitle"]
        )
        if pd.isna(nar):
            equip = _classify_equipment(source_text, SOURCE_FALLBACK_PATTERNS)
            rows.append({
                "nlp_equipment": equip,
                "nlp_assignment_method": "osha_source_fallback" if equip != "Other/Unknown" else "none",
                "error_type": "ambiguous",
                "environmental_mention": 0,
            })
            continue
        s = str(nar).lower()
        equip = _classify_equipment(s, EQUIPMENT_PATTERNS)
        method = "narrative"
        if equip == "Other/Unknown":
            equip = _classify_equipment(source_text, SOURCE_FALLBACK_PATTERNS)
            method = "osha_source_fallback" if equip != "Other/Unknown" else "none"
        mech = len(re.findall(MECHANICAL_PAT, s))
        oper = len(re.findall(OPERATOR_PAT, s))
        env = len(re.findall(ENV_PAT, s))
        if mech > oper:
            et = "mechanical"
        elif oper > mech:
            et = "operator"
        else:
            et = "ambiguous"
        rows.append({
            "nlp_equipment": equip,
            "nlp_assignment_method": method,
            "error_type": et,
            "environmental_mention": int(env > 0),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def build(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        print(f"Cache found -> {CACHE_PATH} (delete to rebuild)")
        return pd.read_parquet(CACHE_PATH)

    df = load_and_filter_data(FILE_PATH)
    nlp = extract_nlp(df)
    df = pd.concat([df.reset_index(drop=True), nlp.reset_index(drop=True)], axis=1)
    before_drop = len(df)
    df = df[df["nlp_equipment"] != "Other/Unknown"].copy().reset_index(drop=True)
    print(f"  Dropped unresolved Other/Unknown equipment rows: {before_drop - len(df)}")
    print(f"  Modeling cohort before weather retrieval: {len(df)}")
    print(f"  NLP assignment methods: {df['nlp_assignment_method'].value_counts().to_dict()}")

    df = add_employer_history(df)
    df = batch_weather(df, max_workers=50)

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
