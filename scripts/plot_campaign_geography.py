#!/usr/bin/env python3
"""
Campaign Geography & Flight-Track Figures
============================================
Four paper-ready figures built from L1's GPS/altitude data:

  campaign_location_map.png          - median location of each of the 12
                                        campaigns: a main CONUS-region
                                        panel (extended slightly north/east
                                        to include AIRS-II/Ottawa) plus a
                                        small Alaska inset (ISDAC, MPACE)
                                        and a small Western Pacific/Guam
                                        inset (ATTREX).
  campaign_flight_trajectories.png   - 4x3 grid, one panel per campaign,
                                        Lat/Lon point cloud of that
                                        campaign's own L1 rows, axes zoomed
                                        to that campaign's own extent. Every
                                        panel's extent is normalized to one
                                        shared aspect ratio (PANEL_ASPECT),
                                        so all 12 render at identical
                                        physical size -- a campaign whose
                                        data box under-fills that ratio gets
                                        extra geographic context on its
                                        shorter side, never a stretched map.
  campaign_altitude_timeseries.png   - 4x3 grid, one panel per campaign,
                                        Alt_m vs Timestamp. Each panel's
                                        x-axis is necessarily per-campaign
                                        (different years); the y-axis
                                        (altitude) range is IDENTICAL
                                        across all 12 panels, computed from
                                        L1's actual min/max, so altitude is
                                        directly comparable panel to panel.
  campaign_altitude_boxplot.png      - single-axes box-and-whisker of
                                        Alt_m, one box per campaign, same
                                        order/color as the other figures.

Style: a 6-color Okabe-Ito (colorblind-safe) palette, each color used by
two campaigns distinguished by line style / marker shape (solid+circle vs.
dashed+triangle) -- the same color+style assignment is used identically
across all three figures for a given campaign.

Outputs: figs/plot_campaign_geography/<timestamp>/*.png, with a `latest`
symlink kept pointing at the newest run.

Usage:
    python scripts/plot_campaign_geography.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
from shapely.geometry import box as _shapely_box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.log_paths import timestamp as _run_timestamp, update_latest  # noqa: E402

# ---------------------------------------------------------------------------
# Shared style: 6-color Okabe-Ito colorblind-safe palette, each color used
# twice (solid/circle vs. dashed/triangle) to cover all 12 campaigns.
# ---------------------------------------------------------------------------
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#0072B2",  # blue
]

CAMPAIGN_ORDER = [
    "ARM", "AIRS-II", "ATTREX", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND",
    "ICE-L", "IPHEX", "ISDAC", "MACPEX", "MC3E", "MIDCIX", "MPACE",
]

CAMPAIGN_STYLE: dict[str, dict] = {}
for i, camp in enumerate(CAMPAIGN_ORDER):
    color = OKABE_ITO[i % 6]
    variant_b = i >= 6
    CAMPAIGN_STYLE[camp] = {
        "color": color,
        "linestyle": "--" if variant_b else "-",
        "marker": "^" if variant_b else "o",
    }

# Campaigns grouped by which map panel they belong to.
MAIN_PANEL_CAMPAIGNS = [
    "ARM", "AIRS-II", "CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND",
    "ICE-L", "IPHEX", "MACPEX", "MC3E", "MIDCIX",
]
ALASKA_INSET_CAMPAIGNS = ["ISDAC", "MPACE"]
PACIFIC_INSET_CAMPAIGNS = ["ATTREX"]

# ---------------------------------------------------------------------------
# Plot 2 (trajectory panels): geographic-context anchors, one per campaign.
# (lat, lon, label). Sources: ARM SGP Central Facility (well-established ARM
# site coordinate); Ottawa and Asheville/Ellington Field established from
# this dataset's own campaign-summary research; Andersen AFB Guam and Rocky
# Mountain Metro Airport (ICE-L's NSF/NCAR C-130 home base) confirmed via
# web search. CRYSTAL-FACE (both NASA and UND aircraft) operated out of
# Key West. ISDAC/MPACE both flew from Utqiagvik (Barrow), AK.
# ---------------------------------------------------------------------------
CAMPAIGN_ANCHOR: dict[str, tuple[float, float, str]] = {
    "ARM": (36.605, -97.485, "SGP Central Facility"),
    "MC3E": (36.605, -97.485, "SGP Central Facility"),
    "MIDCIX": (36.605, -97.485, "SGP Central Facility"),
    "AIRS-II": (45.4215, -75.6972, "Ottawa"),
    "ATTREX": (13.5844, 144.9300, "Andersen AFB, Guam"),
    "CRYSTAL-FACE-NASA": (24.5551, -81.7800, "Key West"),
    "CRYSTAL-FACE-UND": (24.5551, -81.7800, "Key West"),
    "ICE-L": (39.9088, -105.1172, "Rocky Mtn Metro Airport"),
    "IPHEX": (35.5951, -82.5515, "Asheville"),
    "MACPEX": (29.6073, -95.1589, "Ellington Field, Houston"),
    "ISDAC": (71.2906, -156.7887, "Utqiagvik (Barrow)"),
    "MPACE": (71.2906, -156.7887, "Utqiagvik (Barrow)"),
}

# Countries to label when visible (skip USA/Canada -- already implied by
# state/province labels, labeling them too would be redundant clutter).
ALLOWED_COUNTRY_NAMES = {"Cuba", "Bahamas", "Mexico", "Guam", "Northern Mariana Islands",
                          "Philippines", "Palau"}

# Per-campaign country-label whitelist, same mechanism as
# MARINE_LABEL_INCLUDE below (default None = show everything in
# ALLOWED_COUNTRY_NAMES found in the panel's extent).
COUNTRY_LABEL_INCLUDE: dict[str, set[str]] = {
    "_OVERVIEW_PACIFIC_INSET": {"Guam"},
}

# Per-campaign label curation for Plot 2 (trajectory panels). Default
# behavior (a campaign not listed below) is unchanged: show every
# state/country/marine label the generic Natural-Earth lookup finds. These
# are targeted overrides only, requested after visually reviewing the
# rendered figure.
MARINE_LABEL_INCLUDE: dict[str, set[str]] = {
    "ATTREX": {"North Pacific Ocean"},
    "CRYSTAL-FACE-NASA": {"Gulf of Mexico"},
    "CRYSTAL-FACE-UND": {"Gulf of Mexico"},
    "IPHEX": {"North Atlantic Ocean"},
    "ISDAC": {"Arctic Ocean"},
    "MPACE": {"Arctic Ocean"},
    "_OVERVIEW_PACIFIC_INSET": {"North Pacific Ocean"},
}
MARINE_NAME_DISPLAY_OVERRIDE = {"North Atlantic Ocean": "Atlantic Ocean"}

# (campaign, name) -> (dlon, dlat) display-only nudge, so a specific
# label clears a nearby anchor star/other label once actually rendered.
COUNTRY_LABEL_OFFSET: dict[tuple[str, str], tuple[float, float]] = {
    ("ATTREX", "Guam"): (2.2, -2.0),
}
MARINE_LABEL_OFFSET: dict[tuple[str, str], tuple[float, float]] = {
    # Nudged east off the left spine -- the clipped Arctic Ocean polygon's
    # representative point lands near the panel's west edge, where the
    # label would sit on top of the latitude tick labels.
    ("ISDAC", "Arctic Ocean"): (1.5, 0.0),
    ("MPACE", "Arctic Ocean"): (1.5, 0.0),
}
STATE_LABEL_OFFSET: dict[tuple[str, str], tuple[float, float]] = {
    ("ARM", "OK"): (0.0, 0.15),
    ("MC3E", "OK"): (0.0, 0.15),
}

# campaign -> (dlon, dlat, ha, va) override for the anchor-star text label
# (default: no offset, "  " left-padded text right of the star). ARM/MC3E's
# dense trajectory scatter sits right where the default label would draw,
# so their label moves up-left of the star instead.
ANCHOR_LABEL_POSITION: dict[str, tuple[float, float, str, str]] = {
    "ARM": (-0.05, 0.05, "right", "bottom"),
    "MC3E": (-0.05, 0.05, "right", "bottom"),
}

# Campaigns whose anchor star is drawn but whose text label is suppressed
# (e.g. redundant once a nearby city/state label was added instead).
ANCHOR_LABEL_HIDDEN: set[str] = {"CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND"}

# State labels the generic centroid-based lookup can't find because the
# state's Natural-Earth-precomputed label point sits outside the panel's
# zoomed extent even though a sliver of the state is visible in it.
EXTRA_STATE_LABELS: dict[str, list[tuple[str, float, float]]] = {
    "ICE-L": [("CO", 40.3, -107.0)],
    "CRYSTAL-FACE-NASA": [("FL", 26.6, -81.3)],
    "CRYSTAL-FACE-UND": [("FL", 26.6, -81.3)],
}

# Major-city orientation markers (name, lat, lon), separate from
# CAMPAIGN_ANCHOR (that's the campaign's own operating base). Rendered as
# a small black dot + text, distinct from the anchor's star marker. The
# panel extent is expanded (see _expand_extent_to_include) if needed so
# the city is actually visible.
EXTRA_CITY_LABELS: dict[str, list[tuple[str, float, float]]] = {
    "CRYSTAL-FACE-NASA": [("Tampa", 27.9506, -82.4572)],
    "CRYSTAL-FACE-UND": [("Tampa", 27.9506, -82.4572)],
}

# Plot 1 (overview map) only: deliberate display-only lon/lat nudge for
# markers that would otherwise sit exactly on top of each other at that
# map's zoom level. Does not affect any other figure or the underlying
# data -- purely a marker-position schematic offset, disclosed in the
# figure's caption note.
LOCATION_MAP_OFFSET: dict[str, tuple[float, float]] = {
    "CRYSTAL-FACE-NASA": (-0.45, -0.35),   # (dlon, dlat)
    "CRYSTAL-FACE-UND": (0.45, 0.35),
    "ISDAC": (-2.2, 0.8),
    "MPACE": (2.2, -0.8),
}

# Plot 2 (trajectory panels) only: campaign pairs whose maps are close
# enough to be near-duplicates -- render both members of a pair with the
# EXACT same extent (and therefore identical state/country/marine labels,
# since those are derived from the extent) rather than each computing its
# own, so a reader can directly compare the two panels' trajectories on
# one consistent map.
SHARED_EXTENT_GROUPS: list[tuple[str, ...]] = [
    ("ARM", "MC3E"),
    ("CRYSTAL-FACE-NASA", "CRYSTAL-FACE-UND"),
    ("ISDAC", "MPACE"),
]

# Plot 2 (trajectory panels) only: target panel aspect ratio, expressed as
# (longitude span) / (latitude span) in DEGREES. These panels are drawn on
# ccrs.PlateCarree(), where 1 deg lon and 1 deg lat render at the same
# physical size, so this degree ratio IS the rendered physical aspect ratio
# -- deliberately no cos(lat) correction. A cos(lat) factor used to be
# applied here, and it is what made the high-latitude ISDAC/MPACE panels
# render as ~3:1 strips while the low-latitude CRYSTAL-FACE panels rendered
# near-square, despite the code intending all 12 to match.
PANEL_ASPECT = 1.40

STYLE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
}


def _parse_args() -> argparse.Namespace:
    ts = _run_timestamp()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l1", type=Path,
                   default=ROOT / "data" / "out" / "combined_env_data_L1.parquet")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figs" / "plot_campaign_geography" / ts)
    return p.parse_args()


def plot_location_map(df: pd.DataFrame, out_path: Path) -> None:
    medians = df.groupby("Campaign")[["Lat", "Lon"]].median()

    fig = plt.figure(figsize=(11, 7.5))

    # --- Main panel: CONUS + southern Canada margin (covers AIRS-II/Ottawa) ---
    ax_main = fig.add_axes([0.30, 0.08, 0.68, 0.84], projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=39))
    ax_main.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax_main.add_feature(cfeature.LAND, facecolor="#f0f0ec", zorder=0)
    ax_main.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
    ax_main.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax_main.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=1)
    ax_main.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4,
                         edgecolor="gray", zorder=1)
    ax_main.set_title("CONUS", fontsize=10)

    for camp in MAIN_PANEL_CAMPAIGNS:
        lat, lon = medians.loc[camp, "Lat"], medians.loc[camp, "Lon"]
        if camp in LOCATION_MAP_OFFSET:
            dlon, dlat = LOCATION_MAP_OFFSET[camp]
            lon, lat = lon + dlon, lat + dlat
        s = CAMPAIGN_STYLE[camp]
        ax_main.plot(lon, lat, marker=s["marker"], color=s["color"],
                     markersize=11, markeredgecolor="black", markeredgewidth=0.6,
                     transform=ccrs.PlateCarree(), zorder=5, linestyle="none")

    # --- Alaska inset ---
    ax_ak = fig.add_axes([0.02, 0.55, 0.26, 0.38], projection=ccrs.LambertConformal(
        central_longitude=-155, central_latitude=65))
    ax_ak.set_extent([-170, -140, 55, 75], crs=ccrs.PlateCarree())
    ax_ak.add_feature(cfeature.LAND, facecolor="#f0f0ec", zorder=0)
    ax_ak.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
    ax_ak.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax_ak.set_title("Alaska", fontsize=9)
    for camp in ALASKA_INSET_CAMPAIGNS:
        lat, lon = medians.loc[camp, "Lat"], medians.loc[camp, "Lon"]
        if camp in LOCATION_MAP_OFFSET:
            dlon, dlat = LOCATION_MAP_OFFSET[camp]
            lon, lat = lon + dlon, lat + dlat
        s = CAMPAIGN_STYLE[camp]
        ax_ak.plot(lon, lat, marker=s["marker"], color=s["color"],
                   markersize=10, markeredgecolor="black", markeredgewidth=0.6,
                   transform=ccrs.PlateCarree(), zorder=5, linestyle="none")

    # --- Western Pacific / Guam inset ---
    ax_pac = fig.add_axes([0.02, 0.08, 0.26, 0.38], projection=ccrs.PlateCarree(
        central_longitude=150))
    ax_pac.set_extent([130, 165, 5, 30], crs=ccrs.PlateCarree())
    ax_pac.add_feature(cfeature.LAND, facecolor="#f0f0ec", zorder=0)
    ax_pac.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
    ax_pac.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax_pac.set_title("Western Pacific", fontsize=9)
    _pac_extent = [130, 165, 5, 30]
    # Distinct dummy campaign key (matches no entry in the Plot-2-specific
    # override dicts) -- this inset keeps its own full label set from the
    # prior iteration, unaffected by ATTREX's new Plot-2 restrictions.
    _add_country_labels(ax_pac, _pac_extent, "_OVERVIEW_PACIFIC_INSET")
    _add_marine_labels(ax_pac, _pac_extent, "_OVERVIEW_PACIFIC_INSET")
    for camp in PACIFIC_INSET_CAMPAIGNS:
        lat, lon = medians.loc[camp, "Lat"], medians.loc[camp, "Lon"]
        s = CAMPAIGN_STYLE[camp]
        ax_pac.plot(lon, lat, marker=s["marker"], color=s["color"],
                    markersize=10, markeredgecolor="black", markeredgewidth=0.6,
                    transform=ccrs.PlateCarree(), zorder=5, linestyle="none")

    # --- Shared legend ---
    handles = [
        plt.Line2D([0], [0], marker=CAMPAIGN_STYLE[c]["marker"], color=CAMPAIGN_STYLE[c]["color"],
                   markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
                   markersize=8, label=c)
        for c in CAMPAIGN_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _normalize_aspect(extent: list[float], target_aspect: float) -> list[float]:
    """Grow `extent` on whichever axis under-fills `target_aspect`, so the
    box ends up at exactly that aspect ratio. Only ever expands, never
    crops -- the under-filled side gains geographic context rather than the
    map being stretched.

    `target_aspect` is (lon span) / (lat span) in DEGREES, which for the
    ccrs.PlateCarree() panels this feeds is the same thing as the rendered
    physical aspect ratio: PlateCarree draws a degree of longitude and a
    degree of latitude at the same physical size, so no cos(lat) correction
    belongs here.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    lon_span, lat_span = lon_max - lon_min, lat_max - lat_min

    current_aspect = lon_span / lat_span if lat_span else target_aspect
    if current_aspect < target_aspect:
        # too tall/narrow -> widen lon span
        extra = (target_aspect * lat_span - lon_span) / 2
        lon_min, lon_max = lon_min - extra, lon_max + extra
    else:
        # too wide/short -> heighten lat span
        extra = (lon_span / target_aspect - lat_span) / 2
        lat_min, lat_max = lat_min - extra, lat_max + extra

    return [lon_min, lon_max, lat_min, lat_max]


def _uniform_extent(lat: pd.Series, lon: pd.Series, target_aspect: float,
                     lo_q: float = 0.02, hi_q: float = 0.98,
                     pad_frac: float = 0.10, min_pad_deg: float = 0.3) -> list[float]:
    """Percentile-based (not min/max) lon/lat bounding box, padded, then
    normalized to `target_aspect` via `_normalize_aspect` so every panel
    ends up the same physical aspect ratio -- angular box gets more context
    on its shorter side, never stretched/distorted.

    Percentile (not min/max) bounds mean a small cluster of transit-leg
    points far from the main flight area (e.g. IPHEX's ~1.7% of rows near
    Grand Forks, ND) doesn't dictate the zoom window; that data is still
    fully present in L1 and any panel not using this function, just not
    driving this panel's crop. Default 2nd/98th (not 1st/99th): IPHEX's
    transit cluster is dense enough (~1.7% of rows) that a 1%-tail cutoff
    still includes part of it -- 2% comfortably excludes it while still
    only trimming a small fraction of any campaign's real data.
    """
    lon_min, lon_max = lon.quantile(lo_q), lon.quantile(hi_q)
    lat_min, lat_max = lat.quantile(lo_q), lat.quantile(hi_q)
    lon_pad = max((lon_max - lon_min) * pad_frac, min_pad_deg)
    lat_pad = max((lat_max - lat_min) * pad_frac, min_pad_deg)
    lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad
    lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad

    return _normalize_aspect([lon_min, lon_max, lat_min, lat_max], target_aspect)


def _expand_extent_to_include(extent: list[float], lat: float, lon: float,
                               target_aspect: float, buffer_deg: float = 0.1) -> list[float]:
    """If (lat, lon) falls outside `extent`, grow the extent just enough to
    include it (plus a small buffer), then re-normalize to `target_aspect`
    so the panel stays undistorted and the same physical size as every
    other panel -- used to guarantee a specific reference point (e.g. an
    orientation city) is visible even if it falls outside the data-driven
    core extent."""
    lon_min, lon_max, lat_min, lat_max = extent
    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
        return extent  # already inside, nothing to do

    lat_min = min(lat_min, lat - buffer_deg)
    lat_max = max(lat_max, lat + buffer_deg)
    lon_min = min(lon_min, lon - buffer_deg)
    lon_max = max(lon_max, lon + buffer_deg)

    return _normalize_aspect([lon_min, lon_max, lat_min, lat_max], target_aspect)


def _label_in_extent(x: float, y: float, extent: list[float]) -> bool:
    lon_min, lon_max, lat_min, lat_max = extent
    return lon_min <= x <= lon_max and lat_min <= y <= lat_max


_STATES_RECORDS = None
_COUNTRIES_RECORDS = None
_MARINE_RECORDS = None


def _get_states_records():
    global _STATES_RECORDS
    if _STATES_RECORDS is None:
        path = shpreader.natural_earth(resolution="50m", category="cultural",
                                        name="admin_1_states_provinces_lakes")
        _STATES_RECORDS = list(shpreader.Reader(path).records())
    return _STATES_RECORDS


def _get_countries_records():
    global _COUNTRIES_RECORDS
    if _COUNTRIES_RECORDS is None:
        path = shpreader.natural_earth(resolution="50m", category="cultural",
                                        name="admin_0_countries")
        _COUNTRIES_RECORDS = list(shpreader.Reader(path).records())
    return _COUNTRIES_RECORDS


def _get_marine_records():
    global _MARINE_RECORDS
    if _MARINE_RECORDS is None:
        path = shpreader.natural_earth(resolution="50m", category="physical",
                                        name="geography_marine_polys")
        _MARINE_RECORDS = list(shpreader.Reader(path).records())
    return _MARINE_RECORDS


def _add_state_labels(ax, extent: list[float], campaign: str) -> None:
    for rec in _get_states_records():
        lon, lat = rec.attributes.get("longitude"), rec.attributes.get("latitude")
        postal = rec.attributes.get("postal")
        if lon is None or lat is None or not postal:
            continue
        if _label_in_extent(lon, lat, extent):
            dlon, dlat = STATE_LABEL_OFFSET.get((campaign, postal), (0.0, 0.0))
            ax.text(lon + dlon, lat + dlat, postal, fontsize=7, fontweight="bold",
                    color="dimgray", ha="center", va="center",
                    transform=ccrs.PlateCarree(), zorder=3)

    for postal, lat, lon in EXTRA_STATE_LABELS.get(campaign, []):
        ax.text(lon, lat, postal, fontsize=7, fontweight="bold",
                color="dimgray", ha="center", va="center",
                transform=ccrs.PlateCarree(), zorder=3)


def _add_country_labels(ax, extent: list[float], campaign: str) -> None:
    include = COUNTRY_LABEL_INCLUDE.get(campaign)
    for rec in _get_countries_records():
        name = rec.attributes.get("NAME")
        if name not in ALLOWED_COUNTRY_NAMES:
            continue
        if include is not None and name not in include:
            continue
        lon, lat = rec.attributes.get("LABEL_X"), rec.attributes.get("LABEL_Y")
        if lon is None or lat is None:
            continue
        if _label_in_extent(lon, lat, extent):
            dlon, dlat = COUNTRY_LABEL_OFFSET.get((campaign, name), (0.0, 0.0))
            ax.text(lon + dlon, lat + dlat, name, fontsize=7, fontstyle="italic",
                    color="dimgray", ha="center", va="center",
                    transform=ccrs.PlateCarree(), zorder=3)


def _add_marine_labels(ax, extent: list[float], campaign: str) -> None:
    """Label named marine polygons (ocean/sea/gulf/strait/bay, ...) that
    actually intersect this panel's extent -- checking geometry
    intersection, not just whether the polygon's own representative
    point falls inside the extent, since large ocean polygons (Pacific
    Ocean, Arctic Ocean, ...) have a representative point far outside
    any zoomed-in panel even though the polygon clearly covers it. The
    label is placed at the representative point of the *intersected*
    (clipped-to-extent) geometry, so it always lands inside the visible
    panel. `MARINE_LABEL_INCLUDE` restricts a campaign to a curated
    subset of names (default: show everything found)."""
    lon_min, lon_max, lat_min, lat_max = extent
    bbox = _shapely_box(lon_min, lat_min, lon_max, lat_max)
    include = MARINE_LABEL_INCLUDE.get(campaign)
    for rec in _get_marine_records():
        name = rec.attributes.get("name")
        if not name:
            continue
        if include is not None and name not in include:
            continue
        try:
            if not rec.geometry.intersects(bbox):
                continue
            inter = rec.geometry.intersection(bbox)
            if inter.is_empty:
                continue
            pt = inter.representative_point()
        except Exception:
            continue
        dlon, dlat = MARINE_LABEL_OFFSET.get((campaign, name), (0.0, 0.0))
        display_name = MARINE_NAME_DISPLAY_OVERRIDE.get(name, name)
        ax.text(pt.x + dlon, pt.y + dlat, display_name, fontsize=6.5, fontstyle="italic",
                color="#4a6fa5", ha="center", va="center",
                transform=ccrs.PlateCarree(), zorder=3)


def plot_flight_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    """4x3 grid, each panel a cartopy GeoAxes zoomed to that campaign's own
    Lat/Lon extent (percentile-based, padded, aspect-normalized so every
    panel renders the same physical size), with coastline/border/state-
    boundary context, state/country/marine-body text labels, and a named
    anchor location -- no gray background or gridlines."""
    # Figure height sized so a 4.0-in-wide (12/3) PANEL_ASPECT axes fits its
    # grid cell with just enough room left for the title and x tick labels.
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.patch.set_facecolor("white")

    # Precompute extents first, so paired campaigns (ARM/MC3E,
    # CRYSTAL-FACE-NASA/UND, ISDAC/MPACE) can share one identical extent
    # -- computed from their POOLED Lat/Lon, not either campaign alone --
    # rather than each computing its own independently.
    grouped_campaigns = {c for group in SHARED_EXTENT_GROUPS for c in group}
    extents: dict[str, list[float]] = {}
    for group in SHARED_EXTENT_GROUPS:
        pooled = df.loc[df["Campaign"].isin(group), ["Lat", "Lon"]].dropna()
        shared_extent = _uniform_extent(pooled["Lat"], pooled["Lon"], PANEL_ASPECT)
        for c in group:
            extents[c] = shared_extent
    for camp in CAMPAIGN_ORDER:
        if camp not in grouped_campaigns:
            sub = df.loc[df["Campaign"] == camp, ["Lat", "Lon"]].dropna()
            extents[camp] = _uniform_extent(sub["Lat"], sub["Lon"], PANEL_ASPECT)

    # Force in any manually-placed extra city/state label point that
    # would otherwise fall outside the data-driven extent (e.g. Tampa
    # for the CRYSTAL-FACE panels).
    for label_dict in (EXTRA_CITY_LABELS, EXTRA_STATE_LABELS):
        for camp, pts in label_dict.items():
            if camp not in extents:
                continue
            for _, lat, lon in pts:
                extents[camp] = _expand_extent_to_include(
                    extents[camp], lat, lon, PANEL_ASPECT)

    # Re-unify: the expansion above ran per campaign, so a group's members
    # could in principle drift apart (they don't today -- both CRYSTAL-FACE
    # panels carry the same Tampa/FL label entries -- but that's incidental).
    # Take the union of each group's expanded extents and re-normalize it,
    # so the paired panels are guaranteed to stay on one identical map.
    for group in SHARED_EXTENT_GROUPS:
        merged = _normalize_aspect([
            min(extents[c][0] for c in group),
            max(extents[c][1] for c in group),
            min(extents[c][2] for c in group),
            max(extents[c][3] for c in group),
        ], PANEL_ASPECT)
        for c in group:
            extents[c] = merged

    for ax, camp in zip(axes.flat, CAMPAIGN_ORDER):
        sub = df.loc[df["Campaign"] == camp, ["Lat", "Lon"]].dropna()
        s = CAMPAIGN_STYLE[camp]

        extent = extents[camp]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#f0f0ec", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3,
                        edgecolor="gray", zorder=1)

        _add_state_labels(ax, extent, camp)
        _add_country_labels(ax, extent, camp)
        _add_marine_labels(ax, extent, camp)

        ax.scatter(sub["Lon"], sub["Lat"], s=1.5, alpha=0.2, color=s["color"],
                   rasterized=True, linewidths=0, transform=ccrs.PlateCarree(),
                   zorder=2)

        anchor_lat, anchor_lon, anchor_label = CAMPAIGN_ANCHOR[camp]
        ax.plot(anchor_lon, anchor_lat, marker="*", color="black",
                markersize=9, transform=ccrs.PlateCarree(), zorder=4,
                linestyle="none")
        if camp not in ANCHOR_LABEL_HIDDEN:
            dlon, dlat, ha, va = ANCHOR_LABEL_POSITION.get(
                camp, (0.0, 0.0, "left", "center"))
            label_text = anchor_label if ha == "right" else f"  {anchor_label}"
            ax.text(anchor_lon + dlon, anchor_lat + dlat, label_text, fontsize=6.5,
                    color="black", ha=ha, va=va,
                    transform=ccrs.PlateCarree(), zorder=4)

        for city_label, city_lat, city_lon in EXTRA_CITY_LABELS.get(camp, []):
            ax.plot(city_lon, city_lat, marker="o", color="black",
                    markersize=3, transform=ccrs.PlateCarree(), zorder=4,
                    linestyle="none")
            ax.text(city_lon, city_lat, f"  {city_label}", fontsize=6.5,
                    color="black", ha="left", va="center",
                    transform=ccrs.PlateCarree(), zorder=4)

        ax.set_title(f"{camp}  (n={len(sub):,})", fontsize=9)

        gl = ax.gridlines(draw_labels=True, linewidth=0, alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        # Cap both axes at ~5 ticks on nice degree steps -- the gridliner's
        # default locator crowds the 6pt labels on the wider panels.
        gl.xlocator = mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        gl.ylocator = mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6}

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_altitude_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    alt = df["Alt_m"].dropna()
    y_min, y_max = alt.min(), alt.max()
    pad = (y_max - y_min) * 0.03
    y_min, y_max = y_min - pad, y_max + pad

    with plt.rc_context(STYLE_RC):
        fig, axes = plt.subplots(4, 3, figsize=(12, 14))

        for ax, camp in zip(axes.flat, CAMPAIGN_ORDER):
            sub = df.loc[df["Campaign"] == camp, ["Timestamp", "Alt_m"]].dropna()
            sub = sub.sort_values("Timestamp")
            s = CAMPAIGN_STYLE[camp]
            ax.scatter(sub["Timestamp"], sub["Alt_m"], s=1.0, alpha=0.15,
                       color=s["color"], rasterized=True, linewidths=0)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"{camp}  (n={len(sub):,})", fontsize=9)
            ax.set_ylabel("Altitude (m)", fontsize=7)
            ax.set_xlabel("Date", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.tick_params(axis="x", rotation=30)

            t_min = mdates.date2num(sub["Timestamp"].min())
            t_max = mdates.date2num(sub["Timestamp"].max())
            ax.xaxis.set_major_locator(mticker.FixedLocator(np.linspace(t_min, t_max, 6)))
            # Sub-day spans need time-of-day too, else evenly-spaced ticks
            # can land on the same calendar day and look like duplicates.
            date_fmt = "%b %d %H:%M" if (t_max - t_min) < 1 else "%b %d"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    print(f"  Saved {out_path}")


def plot_altitude_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """Single-axes box-and-whisker of Alt_m, one box per campaign, ordered
    and colored per CAMPAIGN_ORDER/CAMPAIGN_STYLE for consistency with the
    other 3 figures."""
    with plt.rc_context(STYLE_RC):
        fig, ax = plt.subplots(figsize=(11, 6))

        data = [df.loc[df["Campaign"] == c, "Alt_m"].dropna() for c in CAMPAIGN_ORDER]
        bp = ax.boxplot(
            data, tick_labels=CAMPAIGN_ORDER, patch_artist=True, showfliers=False,
            widths=0.6, medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, camp in zip(bp["boxes"], CAMPAIGN_ORDER):
            s = CAMPAIGN_STYLE[camp]
            patch.set_facecolor(s["color"])
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        ax.set_ylabel("Altitude (m)")
        ax.set_xlabel("Campaign")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading L1 from {args.l1} ...")
    df = pd.read_parquet(args.l1, columns=["Campaign", "Lat", "Lon", "Alt_m", "Timestamp"])
    df = df[df["Campaign"].isin(CAMPAIGN_ORDER)]
    print(f"  {len(df):,} rows, {df['Campaign'].nunique()} campaigns")

    missing = set(CAMPAIGN_ORDER) - set(df["Campaign"].unique())
    if missing:
        raise RuntimeError(f"Expected all 12 campaigns in L1, missing: {missing}")

    print("\nPlot 1: location map ...")
    plot_location_map(df, args.out / "campaign_location_map.png")

    print("\nPlot 2: flight trajectories ...")
    plot_flight_trajectories(df, args.out / "campaign_flight_trajectories.png")

    print("\nPlot 3: altitude time series ...")
    plot_altitude_timeseries(df, args.out / "campaign_altitude_timeseries.png")

    print("\nPlot 4: altitude boxplot ...")
    plot_altitude_boxplot(df, args.out / "campaign_altitude_boxplot.png")

    update_latest(args.out.parent, args.out)
    print(f"\nLatest run: {args.out.parent / 'latest'} -> {args.out.name}")


if __name__ == "__main__":
    main()
