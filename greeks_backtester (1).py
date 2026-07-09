"""
╔══════════════════════════════════════════════════════════════════════════╗
║   NIFTY Multi-Strategy Greek-Based Options Backtester v3.0             ║
║   ── with HMM + GMM Regime Detection ──                                ║
║                                                                        ║
║   Strategies:                                                          ║
║     1. Iron Condor      (range-bound regime only)                      ║
║     2. Credit Put Spread (low-vol trending regime only)                ║
║     3. Long Straddle     (regime transition → high-vol)                ║
║     4. Bull Put Spread   (mean-reversion after dips, regime-filtered)  ║
║                                                                        ║
║   Regime Detection:                                                    ║
║     • GMM — discovers natural market clusters (returns + vol + RSI)    ║
║     • HMM — models regime transitions with temporal persistence        ║
║     • Walk-forward — re-fit quarterly, no look-ahead bias              ║
║                                                                        ║
║   Data     : yfinance (spot) + Black-Scholes IV simulation             ║
║   Based on : IIT-K Options Greeks PS Framework                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, json, datetime

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False
    print("⚠️  hmmlearn not found — falling back to GMM-only regime detection.")

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.2f}'.format)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CFG = {
    # ── Backtest window ──────────────────────────────────────
    'FROM_DATE'          : '2022-01-01',
    'TO_DATE'            : '2024-12-31',

    # ── Capital ──────────────────────────────────────────────
    'INITIAL_CAPITAL'    : 500_000,
    'LOT_SIZE'           : 50,
    'MAX_LOTS'           : 2,
    'CAPITAL_RISK_PCT'   : 0.08,

    # ── Options model ────────────────────────────────────────
    'RISK_FREE_RATE'     : 0.065,
    'IV_PREMIUM'         : 0.03,

    # ── Entry timing ─────────────────────────────────────────
    'ENTRY_DTE'          : 30,
    'STRIKE_ROUND'       : 50,

    # ═════════════════════════════════════════════════════════
    # HMM / GMM REGIME DETECTION
    # ═════════════════════════════════════════════════════════
    'HMM_N_REGIMES'           : 3,
    'HMM_TRAIN_WINDOW'        : 200,       # trading days for training
    'HMM_REFIT_INTERVAL'      : 63,        # re-fit quarterly
    'HMM_CONTEXT_DAYS'        : 40,        # HMM context for prediction

    # ── Regime-based entry filters ────────────────────────────
    'REGIME_IC_CONFIDENCE'    : 0.55,      # IC needs range-bound confidence
    'REGIME_IC_MIN_PERSIST'   : 3,         # days in regime before entering
    'REGIME_CPS_CONFIDENCE'   : 0.50,      # CPS needs trending confidence
    'REGIME_CPS_MIN_PERSIST'  : 2,
    'REGIME_LS_HIGHVOL_PROB'  : 0.30,      # LS enters when high-vol prob rises
    'REGIME_BPS_HIGHVOL_SKIP' : 0.60,      # BPS skips when high-vol > 60%

    # ── Dynamic adjustment factors ─────────────────────────────
    'DYN_STOP_VOL_FACTOR'     : 0.25,      # regime vol impact on stops
    'DYN_SIZE_TRANSITION'     : 0.5,       # reduce size during transitions

    # ═════════════════════════════════════════════════════════
    # STRATEGY 1: IRON CONDOR (regime-gated)
    # ═════════════════════════════════════════════════════════
    'IC_IV_HV_MIN'       : 1.10,           # slightly relaxed (regime does the filtering)
    'IC_CALL_OFFSET'     : 8,
    'IC_PUT_OFFSET'      : 8,
    'IC_WING_WIDTH'      : 3,
    'IC_PROFIT_TARGET'   : 0.45,           # 45% profit target — take profits faster
    'IC_STOP_LOSS_MULT'  : 2.0,            # wider stop — let regime filter prevent bad entries
    'IC_MIN_DTE'         : 7,
    'IC_MAX_5D_MOVE'     : 0.035,          # relaxed — regime does the real filtering

    # ═════════════════════════════════════════════════════════
    # STRATEGY 2: CREDIT PUT SPREAD (regime-gated)
    # ═════════════════════════════════════════════════════════
    'CPS_IV_HV_MIN'      : 1.05,           # relaxed
    'CPS_PUT_OFFSET'     : 7,              # wider strikes for safety
    'CPS_WING_WIDTH'     : 3,
    'CPS_PROFIT_TARGET'  : 0.50,
    'CPS_STOP_LOSS_MULT' : 2.5,            # wider stop
    'CPS_MIN_DTE'        : 5,
    'CPS_SMA_BUFFER'     : 0.01,           # relaxed — regime handles filtering

    # ═════════════════════════════════════════════════════════
    # STRATEGY 3: LONG STRADDLE (regime-enhanced)
    # ═════════════════════════════════════════════════════════
    'LS_IV_HV_MAX'       : 1.25,           # more relaxed — regime finds entries
    'LS_PROFIT_TARGET'   : 0.40,
    'LS_STOP_LOSS_PCT'   : 0.15,           # tighter stop: 15% max loss
    'LS_IV_CRUSH_EXIT'   : 0.08,           # faster IV crush exit
    'LS_MIN_DTE'         : 7,
    'LS_MAX_HOLD_DAYS'   : 8,              # shorter hold — less theta decay

    # ═════════════════════════════════════════════════════════
    # STRATEGY 4: BULL PUT SPREAD ON DIPS (regime-filtered)
    # ═════════════════════════════════════════════════════════
    'BPS_DIP_THRESHOLD'  : -0.025,
    'BPS_DIP_LOOKBACK'   : 5,
    'BPS_PUT_OFFSET'     : 5,              # wider for safety
    'BPS_WING_WIDTH'     : 3,
    'BPS_ENTRY_DTE'      : 15,
    'BPS_PROFIT_TARGET'  : 0.45,           # faster profit-taking
    'BPS_STOP_LOSS_MULT' : 3.0,            # wider stop — mean reversion needs room
    'BPS_MIN_DTE'        : 3,
    'BPS_COOLDOWN_DAYS'  : 7,

    # ── Greek thresholds (shared) ─────────────────────────────
    'DELTA_ALERT'        : 0.25,
    'DELTA_HARD_EXIT'    : 0.40,
    'GAMMA_ALERT'        : 0.0015,
    'GAMMA_EXIT'         : 0.0020,
    'VEGA_IV_ALERT_PCT'  : 0.25,
    'VEGA_IV_EXIT_PCT'   : 0.45,
}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — BLACK-SCHOLES ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _bs_price(S, K, T, r, sigma, opt):
    """Raw B-S option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if opt == 'CE' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def greeks(S, K, T, r, sigma, opt):
    """
    Returns dict with: price, delta, gamma, vega, theta, iv
    vega  = per 1% change in IV
    theta = per calendar day
    """
    if T <= 1e-6 or sigma <= 1e-6:
        return dict(price=0, delta=0, gamma=0, vega=0, theta=0, iv=sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price  = _bs_price(S, K, T, r, sigma, opt)
    delta  = norm.cdf(d1) if opt == 'CE' else norm.cdf(d1) - 1
    gamma  = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega   = S * norm.pdf(d1) * np.sqrt(T) / 100
    th_raw = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if opt == 'CE':
        theta = (th_raw - r * K * np.exp(-r * T) * norm.cdf(d2))  / 365
    else:
        theta = (th_raw + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return dict(price=max(price, 0), delta=delta, gamma=gamma,
                vega=vega, theta=theta, iv=sigma)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def compute_rsi(prices, period=14):
    """Relative Strength Index — momentum oscillator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_regime_features(df):
    """
    Add enhanced features for GMM/HMM regime detection.
    All features are backward-looking (no look-ahead bias).
    """
    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    # ── Core features ────────────────────────────────────────
    df['log_ret'] = log_ret

    # RSI (14-period)
    df['RSI14'] = compute_rsi(df['Close'], 14)

    # Bollinger Band width (normalized volatility)
    bb_std = df['Close'].rolling(20).std()
    df['BB_width'] = (2 * bb_std) / df['SMA20']

    # Volume Z-score (is volume unusual?)
    vol_mean = df['Volume'].rolling(50).mean()
    vol_std  = df['Volume'].rolling(50).std().replace(0, 1)
    df['vol_zscore'] = (df['Volume'] - vol_mean) / vol_std

    # Realized skewness (10-day rolling) — crash/rally indicator
    df['skew_10d'] = log_ret.rolling(10).apply(
        lambda x: pd.Series(x).skew() if len(x) > 2 else 0, raw=False
    )

    # Mean-reversion Z-score (price vs 20-day mean / std)
    df['mean_rev_zscore'] = (df['Close'] - df['SMA20']) / bb_std.replace(0, 1)

    # ── Clean NaN ────────────────────────────────────────────
    feat_cols = ['log_ret', 'RSI14', 'BB_width', 'vol_zscore',
                 'skew_10d', 'mean_rev_zscore']
    for col in feat_cols:
        df[col] = df[col].bfill().fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — REGIME DETECTION (GMM + HMM)
# ═══════════════════════════════════════════════════════════════════════════

# Feature columns used for regime clustering
REGIME_FEATURES = ['log_ret', 'HV20', 'IV_HV_ratio', 'RSI14', 'vol_zscore', 'BB_width']

# Regime labels (canonical)
REGIME_TRENDING   = 0   # Low-vol trending (calm uptrend or gentle downtrend)
REGIME_RANGE      = 1   # Range-bound / choppy
REGIME_HIGH_VOL   = 2   # High-vol stress (crashes, sharp rallies)
REGIME_UNKNOWN    = -1   # Not yet predicted (training period)

REGIME_NAMES = {
    REGIME_TRENDING : 'LOW_VOL_TRENDING',
    REGIME_RANGE    : 'RANGE_BOUND',
    REGIME_HIGH_VOL : 'HIGH_VOL_STRESS',
    REGIME_UNKNOWN  : 'UNKNOWN',
}


class RegimeDetector:
    """
    GMM + HMM regime detection.
    GMM discovers natural clusters; HMM adds temporal persistence.
    """

    def __init__(self, n_regimes=3):
        self.n_regimes  = n_regimes
        self.gmm        = None
        self.hmm        = None
        self.scaler     = StandardScaler()
        self.regime_order = None   # maps fitted labels → canonical labels
        self.fitted     = False

    def _extract_features(self, df):
        """Get feature matrix from DataFrame."""
        X = df[REGIME_FEATURES].values.copy()
        # Replace inf/nan with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, df):
        """Fit GMM and HMM on historical data."""
        X_raw = self._extract_features(df)
        mask  = np.all(np.isfinite(X_raw), axis=1)
        X_clean = X_raw[mask]

        if len(X_clean) < 50:
            return False

        X_scaled = self.scaler.fit_transform(X_clean)

        # ── Step 1: GMM clustering ──────────────────────────────
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=15,
            random_state=42,
            max_iter=300,
            reg_covar=1e-4,
        )
        self.gmm.fit(X_scaled)
        gmm_labels = self.gmm.predict(X_scaled)

        # ── Step 2: HMM (temporal dependence) ───────────────────
        if HAS_HMMLEARN:
            self.hmm = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=150,
                random_state=42,
                tol=1e-4,
            )
            # Initialize HMM with GMM parameters for faster convergence
            self.hmm.means_init  = self.gmm.means_.copy()
            self.hmm.covars_prior = self.gmm.covariances_.copy()
            try:
                self.hmm.fit(X_scaled)
            except Exception:
                # If HMM fitting fails, fall back to GMM
                self.hmm = None

        # ── Step 3: Map regime labels by volatility ─────────────
        # Sort regimes by mean HV20 (column index 1): lowest vol → TRENDING,
        # mid → RANGE_BOUND, highest → HIGH_VOL_STRESS
        model = self.hmm if (self.hmm is not None) else self.gmm
        if hasattr(model, 'predict'):
            labels = model.predict(X_scaled)
        else:
            labels = gmm_labels

        regime_vols = {}
        for i in range(self.n_regimes):
            mask_i = labels == i
            if mask_i.sum() > 0:
                regime_vols[i] = X_clean[mask_i, 1].mean()  # HV20 is col 1
            else:
                regime_vols[i] = float(i)

        sorted_by_vol = sorted(regime_vols.keys(), key=lambda x: regime_vols[x])
        self.regime_order = {sorted_by_vol[i]: i for i in range(self.n_regimes)}

        self.fitted = True
        return True

    def predict(self, df):
        """Predict regime labels, probabilities, and confidence for each day."""
        if not self.fitted:
            return None, None, None

        X_raw = self._extract_features(df)
        X_scaled = self.scaler.transform(X_raw)

        model = self.hmm if (self.hmm is not None) else self.gmm

        try:
            raw_labels = model.predict(X_scaled)
            raw_probs  = model.predict_proba(X_scaled)
        except Exception:
            raw_labels = self.gmm.predict(X_scaled)
            raw_probs  = self.gmm.predict_proba(X_scaled)

        # Remap labels and probabilities to canonical order
        mapped_labels = np.array([self.regime_order.get(l, l) for l in raw_labels])

        mapped_probs = np.zeros_like(raw_probs)
        for old_lbl, new_lbl in self.regime_order.items():
            if old_lbl < raw_probs.shape[1]:
                mapped_probs[:, new_lbl] = raw_probs[:, old_lbl]

        confidence = mapped_probs.max(axis=1)

        return mapped_labels, mapped_probs, confidence


def run_walk_forward_regime_detection(df):
    """
    Walk-forward HMM/GMM regime detection.
    Trains on a rolling window, predicts forward — no look-ahead bias.
    """
    train_window   = CFG['HMM_TRAIN_WINDOW']
    refit_interval = CFG['HMM_REFIT_INTERVAL']
    context_days   = CFG['HMM_CONTEXT_DAYS']
    n_regimes      = CFG['HMM_N_REGIMES']
    n = len(df)

    # Initialize regime columns with defaults
    df['regime']            = REGIME_UNKNOWN
    df['regime_confidence'] = 0.0
    df['prob_trending']     = 0.33
    df['prob_range_bound']  = 0.33
    df['prob_high_vol']     = 0.33

    if n < train_window + 10:
        print("    ⚠️  Not enough data for walk-forward regime detection")
        return df

    detector = RegimeDetector(n_regimes=n_regimes)

    # Build list of (train_end, test_end) intervals
    intervals = []
    t = train_window
    while t < n:
        test_end = min(t + refit_interval, n)
        intervals.append((t, test_end))
        t = test_end

    print(f"    🔬  Walk-forward: {len(intervals)} re-fit windows "
          f"(train={train_window}d, refit={refit_interval}d)")

    for train_end, test_end in intervals:
        # Training data: last `train_window` days up to train_end
        train_start = max(0, train_end - train_window)
        train_data  = df.iloc[train_start:train_end]

        success = detector.fit(train_data)
        if not success:
            continue

        # Prediction data: include context before test period for HMM sequencing
        context_start = max(0, train_end - context_days)
        pred_data     = df.iloc[context_start:test_end]

        labels, probs, conf = detector.predict(pred_data)
        if labels is None:
            continue

        # Write predictions only for the test period (train_end → test_end)
        offset = train_end - context_start
        test_len = test_end - train_end

        for j in range(test_len):
            pred_idx = offset + j
            if pred_idx >= len(labels):
                break
            df_idx = train_end + j
            if df_idx >= n:
                break

            iloc_pos = df_idx
            df.iloc[iloc_pos, df.columns.get_loc('regime')]            = int(labels[pred_idx])
            df.iloc[iloc_pos, df.columns.get_loc('regime_confidence')] = float(conf[pred_idx])
            df.iloc[iloc_pos, df.columns.get_loc('prob_trending')]     = float(probs[pred_idx, 0])
            df.iloc[iloc_pos, df.columns.get_loc('prob_range_bound')]  = float(probs[pred_idx, 1])
            df.iloc[iloc_pos, df.columns.get_loc('prob_high_vol')]     = float(probs[pred_idx, 2])

    # ── Compute regime persistence (days in current regime) ──────
    df['regime_persistence'] = 0
    curr_regime = REGIME_UNKNOWN
    count = 0
    for i in range(n):
        r = df.iloc[i]['regime']
        if r == curr_regime:
            count += 1
        else:
            curr_regime = r
            count = 1
        df.iloc[i, df.columns.get_loc('regime_persistence')] = count

    # Regime name
    df['regime_name'] = df['regime'].map(REGIME_NAMES)

    # ── Print regime statistics ──────────────────────────────────
    valid_mask = df['regime'] != REGIME_UNKNOWN
    if valid_mask.sum() > 0:
        regime_counts = df.loc[valid_mask, 'regime'].value_counts()
        total = valid_mask.sum()
        print(f"    📊  Regime distribution ({total} predicted days):")
        for r_val in sorted(regime_counts.index):
            pct = regime_counts[r_val] / total * 100
            name = REGIME_NAMES.get(int(r_val), '?')
            print(f"        {name:22s}: {regime_counts[r_val]:>4d} days ({pct:.1f}%)")
        avg_conf = df.loc[valid_mask, 'regime_confidence'].mean()
        print(f"        Average confidence    : {avg_conf:.2%}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_nifty(from_date: str, to_date: str) -> pd.DataFrame:
    print(f"📥  Downloading NIFTY 50  [{from_date} → {to_date}] ...")
    df = yf.download("^NSEI", start=from_date, end=to_date,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty data. Check dates / internet.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(inplace=True)

    log_ret = np.log(df['Close'] / df['Close'].shift(1))

    # Historical volatilities
    df['HV10'] = log_ret.rolling(10).std() * np.sqrt(252)
    df['HV20'] = log_ret.rolling(20).std() * np.sqrt(252)
    df['HV30'] = log_ret.rolling(30).std() * np.sqrt(252)
    df['HV10'] = df['HV10'].bfill().fillna(0.15)
    df['HV20'] = df['HV20'].bfill().fillna(0.15)
    df['HV30'] = df['HV30'].bfill().fillna(0.15)

    # IV estimate
    df['IV'] = (df['HV20'] + CFG['IV_PREMIUM']).clip(0.10, 0.80)
    df['IV_HV_ratio'] = df['IV'] / df['HV30'].replace(0, 0.15)
    df['vol_gap'] = df['IV'] - df['HV10']

    # Trend indicators
    df['SMA20']  = df['Close'].rolling(20).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA20']  = df['SMA20'].bfill()
    df['SMA50']  = df['SMA50'].bfill()
    df['SMA100'] = df['SMA100'].bfill()

    df['trend']  = np.where(df['SMA50'] > df['SMA100'], 1, -1)
    df['ret_5d'] = df['Close'].pct_change(5)

    # ── Add regime features ──────────────────────────────────
    df = add_regime_features(df)

    print(f"    ✅  {len(df)} trading days loaded")
    print(f"    IV/HV ratio range: "
          f"{df['IV_HV_ratio'].min():.2f} – {df['IV_HV_ratio'].max():.2f}  "
          f"| mean: {df['IV_HV_ratio'].mean():.2f}")
    bullish_pct = (df['trend'] == 1).mean() * 100
    print(f"    Trend: {bullish_pct:.0f}% bullish days | {100-bullish_pct:.0f}% bearish days")

    # ── Run walk-forward regime detection ─────────────────────
    print(f"\n🧠  Running HMM + GMM Regime Detection (walk-forward) ...")
    df = run_walk_forward_regime_detection(df)

    return df


def get_monthly_expiry_dates(trading_dates: pd.DatetimeIndex,
                              from_date: str, to_date: str) -> list:
    """Generate last-Thursday-of-each-month expiry dates that fall on trading days."""
    fd = pd.Timestamp(from_date)
    td = pd.Timestamp(to_date)
    expiries = []

    current = fd.to_period('M')
    end_per = td.to_period('M')

    while current <= end_per:
        year  = current.year
        month = current.month
        last_day = pd.Timestamp(year=year, month=month,
                                day=pd.Timestamp(year, month, 1).days_in_month)
        while last_day.weekday() != 3:
            last_day -= pd.Timedelta(days=1)

        cand = last_day
        for _ in range(5):
            if cand in trading_dates:
                expiries.append(cand)
                break
            cand -= pd.Timedelta(days=1)

        current += 1

    return sorted(set(expiries))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — UTILITY: DYNAMIC RISK PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

def dynamic_stop_loss(base_mult, entry_row, df_hv_mean):
    """Adjust stop-loss based on current regime volatility."""
    hv_now  = float(entry_row.get('HV20', 0.15))
    vol_factor = hv_now / max(df_hv_mean, 0.01)
    adjusted = base_mult * (1 + CFG['DYN_STOP_VOL_FACTOR'] * max(vol_factor - 1, 0))
    return min(adjusted, base_mult * 2.0)   # cap at 2× base


def dynamic_lots(base_lots, entry_row):
    """Reduce lots during regime transitions (low confidence)."""
    conf = float(entry_row.get('regime_confidence', 1.0))
    if conf < 0.55:
        return max(1, int(base_lots * CFG['DYN_SIZE_TRANSITION']))
    return base_lots


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — STRATEGY BACKTESTERS
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# 7A — IRON CONDOR (regime-gated: RANGE_BOUND only)
# ─────────────────────────────────────────────────────────────────────────

class IronCondorBacktester:
    """
    Iron Condor — only enters when HMM regime = RANGE_BOUND
    with sufficient confidence and persistence.
    """

    def __init__(self, df: pd.DataFrame, cash: float = None):
        self.df     = df
        self.r      = CFG['RISK_FREE_RATE']
        self.cash   = cash if cash is not None else CFG['INITIAL_CAPITAL']
        self.trades = []
        self.alerts = []
        self.pv_log = []
        self._tid   = 0
        self._hv_mean = df['HV20'].mean()

    def run(self, from_date: str, to_date: str):
        expiries = get_monthly_expiry_dates(self.df.index, from_date, to_date)
        print(f"\n📅  IC: {len(expiries)} monthly expiry cycles\n")
        for exp in expiries:
            self._run_cycle(exp)
        print(f"    ✅  Iron Condor — {len(self.trades)} trades")
        return self.trades, self.alerts

    def _run_cycle(self, expiry: pd.Timestamp):
        all_dates  = self.df.index
        pre_expiry = all_dates[all_dates < expiry]
        if len(pre_expiry) < CFG['ENTRY_DTE'] + 5:
            return

        entry_date = pre_expiry[-CFG['ENTRY_DTE']]
        entry_row  = self.df.loc[entry_date]
        S0         = float(entry_row['Close'])
        iv0        = float(entry_row['IV'])
        iv_hv      = float(entry_row['IV_HV_ratio'])
        ret_5d     = float(entry_row.get('ret_5d', 0))

        # ── REGIME FILTER (PRIMARY GATE) ─────────────────────────
        regime      = int(entry_row.get('regime', REGIME_UNKNOWN))
        regime_conf = float(entry_row.get('regime_confidence', 0))
        regime_pers = int(entry_row.get('regime_persistence', 0))
        prob_range  = float(entry_row.get('prob_range_bound', 0))
        prob_highv  = float(entry_row.get('prob_high_vol', 0))

        # Only enter in RANGE_BOUND regime with sufficient confidence
        if regime != REGIME_UNKNOWN:  # skip filter during training period
            if regime != REGIME_RANGE:
                return
            if prob_range < CFG['REGIME_IC_CONFIDENCE']:
                return
            if regime_pers < CFG['REGIME_IC_MIN_PERSIST']:
                return
            # Abort if high-vol probability is creeping up
            if prob_highv > 0.35:
                return

        # ── Existing filters (relaxed — regime is primary) ───────
        if iv_hv < CFG['IC_IV_HV_MIN']:
            return
        if abs(ret_5d) > CFG['IC_MAX_5D_MOVE']:
            return

        step = CFG['STRIKE_ROUND']
        ATM  = round(S0 / step) * step

        # Dynamic strike offset: wider when vol is higher within regime
        vol_adj = max(1.0, float(entry_row['HV20']) / self._hv_mean)
        call_offset = int(CFG['IC_CALL_OFFSET'] * vol_adj)
        put_offset  = int(CFG['IC_PUT_OFFSET']  * vol_adj)

        K_sc = ATM + call_offset * step
        K_lc = K_sc + CFG['IC_WING_WIDTH'] * step
        K_sp = ATM - put_offset  * step
        K_lp = K_sp - CFG['IC_WING_WIDTH'] * step

        T0   = CFG['ENTRY_DTE'] / 365

        sc0  = greeks(S0, K_sc, T0, self.r, iv0, 'CE')
        lc0  = greeks(S0, K_lc, T0, self.r, iv0, 'CE')
        sp0  = greeks(S0, K_sp, T0, self.r, iv0, 'PE')
        lp0  = greeks(S0, K_lp, T0, self.r, iv0, 'PE')

        net_prem0 = (sc0['price'] - lc0['price']) + (sp0['price'] - lp0['price'])
        if net_prem0 <= 0:
            return

        max_loss_per_lot = (CFG['IC_WING_WIDTH'] * step - net_prem0) * CFG['LOT_SIZE']
        risk = self.cash * CFG['CAPITAL_RISK_PCT']
        lots_by_prem = max(1, int(risk / (net_prem0 * CFG['LOT_SIZE'])))
        lots_by_loss = max(1, int(risk / max_loss_per_lot)) if max_loss_per_lot > 0 else 1
        lots = min(lots_by_prem, lots_by_loss, CFG['MAX_LOTS'])

        # Dynamic sizing: reduce during low-confidence transitions
        lots = dynamic_lots(lots, entry_row)

        # Dynamic stop loss
        sl_mult = dynamic_stop_loss(CFG['IC_STOP_LOSS_MULT'], entry_row, self._hv_mean)

        self._tid += 1
        trade = {
            'id': self._tid, 'strategy': 'Iron Condor',
            'expiry': str(expiry.date()), 'entry_date': str(entry_date.date()),
            'entry_spot': round(S0, 2), 'ATM': ATM,
            'K_short_call': K_sc, 'K_long_call': K_lc,
            'K_short_put': K_sp, 'K_long_put': K_lp,
            'entry_iv': round(iv0, 4), 'entry_net_premium': round(net_prem0, 2),
            'lots': lots, 'exit_date': None, 'exit_reason': None, 'pnl': None,
            'entry_regime': REGIME_NAMES.get(regime, 'UNKNOWN'),
            'entry_regime_conf': round(regime_conf, 3),
        }

        print(f"  ▶  IC  Entry {entry_date.date()} | Exp {expiry.date()} | "
              f"Spot ₹{S0:.0f} | [{K_lp}/{K_sp}–{K_sc}/{K_lc}] | "
              f"Prem ₹{net_prem0:.1f} | Lots {lots} | "
              f"Regime: {REGIME_NAMES.get(regime,'?')} ({regime_conf:.0%})")

        cycle_dates = all_dates[(all_dates >= entry_date) & (all_dates <= expiry)]
        exited = False

        for day_i, date in enumerate(cycle_dates):
            row     = self.df.loc[date]
            S       = float(row['Close'])
            iv_now  = float(row['IV'])
            dte_rem = len(cycle_dates) - day_i
            T       = max(dte_rem / 365, 1e-6)

            sc_now = greeks(S, K_sc, T, self.r, iv_now, 'CE')
            lc_now = greeks(S, K_lc, T, self.r, iv_now, 'CE')
            sp_now = greeks(S, K_sp, T, self.r, iv_now, 'PE')
            lp_now = greeks(S, K_lp, T, self.r, iv_now, 'PE')

            curr_net_prem = (sc_now['price'] - lc_now['price']) + (sp_now['price'] - lp_now['price'])
            pnl = (net_prem0 - curr_net_prem) * lots * CFG['LOT_SIZE']

            net_delta = -((sc_now['delta'] - lc_now['delta']) +
                          (sp_now['delta'] - lp_now['delta']))
            iv_chg_pct = (iv_now - iv0) / iv0

            profit_target = net_prem0 * lots * CFG['LOT_SIZE'] * CFG['IC_PROFIT_TARGET']
            sl_debit      = net_prem0 * sl_mult

            # ── Regime-based early exit ───────────────────────────
            curr_regime  = int(row.get('regime', REGIME_UNKNOWN))
            curr_prob_hv = float(row.get('prob_high_vol', 0))

            reason = None
            if   day_i == 0                               : reason = None
            elif pnl >= profit_target                     : reason = 'PROFIT_TARGET'
            elif curr_net_prem >= sl_debit                : reason = 'STOP_LOSS'
            elif abs(net_delta) > CFG['DELTA_HARD_EXIT']  : reason = 'DELTA_EXIT'
            elif iv_chg_pct     > CFG['VEGA_IV_EXIT_PCT'] : reason = 'VEGA_EXIT'
            elif dte_rem        <= CFG['IC_MIN_DTE']      : reason = 'TIME_EXIT'
            elif date           == expiry                  : reason = 'EXPIRY'
            # NEW: exit if regime shifts to HIGH_VOL mid-trade
            elif (curr_regime == REGIME_HIGH_VOL and
                  curr_prob_hv > 0.55 and day_i > 2)      : reason = 'REGIME_EXIT'

            if reason:
                self.cash += pnl
                trade.update({
                    'exit_date': str(date.date()), 'exit_spot': round(S, 2),
                    'exit_reason': reason, 'pnl': round(pnl, 2), 'dte_held': day_i,
                    'exit_iv': round(iv_now, 4),
                })
                icon = '✅' if pnl >= 0 else '❌'
                print(f"     {icon} IC  [{reason:<16}] P&L ₹{pnl:>+8,.0f} | DTE:{day_i}")
                exited = True
                break

        if not exited:
            trade['exit_reason'] = 'OPEN_AT_END'
            trade['pnl'] = 0
        self.trades.append(trade)


# ─────────────────────────────────────────────────────────────────────────
# 7B — CREDIT PUT SPREAD (regime-gated: LOW_VOL_TRENDING only)
# ─────────────────────────────────────────────────────────────────────────

class CreditPutSpreadBacktester:
    """
    Credit Put Spread — only enters when HMM regime = LOW_VOL_TRENDING
    and trend is confirmed bullish.
    """

    def __init__(self, df: pd.DataFrame, cash: float = None):
        self.df     = df
        self.r      = CFG['RISK_FREE_RATE']
        self.cash   = cash if cash is not None else CFG['INITIAL_CAPITAL']
        self.trades = []
        self.alerts = []
        self._tid   = 0
        self._hv_mean = df['HV20'].mean()

    def run(self, from_date: str, to_date: str):
        expiries = get_monthly_expiry_dates(self.df.index, from_date, to_date)
        print(f"\n📅  CPS: {len(expiries)} monthly expiry cycles\n")
        for exp in expiries:
            self._run_cycle(exp)
        print(f"    ✅  Credit Put Spread — {len(self.trades)} trades")
        return self.trades, self.alerts

    def _run_cycle(self, expiry: pd.Timestamp):
        all_dates  = self.df.index
        pre_expiry = all_dates[all_dates < expiry]
        if len(pre_expiry) < CFG['ENTRY_DTE'] + 5:
            return

        entry_date = pre_expiry[-CFG['ENTRY_DTE']]
        entry_row  = self.df.loc[entry_date]
        S0         = float(entry_row['Close'])
        iv0        = float(entry_row['IV'])
        iv_hv      = float(entry_row['IV_HV_ratio'])
        trend      = int(entry_row['trend'])
        sma100     = float(entry_row['SMA100'])

        # ── REGIME FILTER (PRIMARY GATE) ─────────────────────────
        regime       = int(entry_row.get('regime', REGIME_UNKNOWN))
        regime_conf  = float(entry_row.get('regime_confidence', 0))
        regime_pers  = int(entry_row.get('regime_persistence', 0))
        prob_trend   = float(entry_row.get('prob_trending', 0))
        prob_highv   = float(entry_row.get('prob_high_vol', 0))

        if regime != REGIME_UNKNOWN:
            # Only enter in LOW_VOL_TRENDING regime
            if regime != REGIME_TRENDING:
                return
            if prob_trend < CFG['REGIME_CPS_CONFIDENCE']:
                return
            if regime_pers < CFG['REGIME_CPS_MIN_PERSIST']:
                return
            # Skip if high-vol probability is rising
            if prob_highv > 0.30:
                return

        # ── Existing filters (relaxed) ───────────────────────────
        if trend != 1 or iv_hv < CFG['CPS_IV_HV_MIN']:
            return
        if S0 < sma100 * (1 + CFG['CPS_SMA_BUFFER']):
            return

        step = CFG['STRIKE_ROUND']
        ATM  = round(S0 / step) * step

        K_sp = ATM - CFG['CPS_PUT_OFFSET'] * step
        K_lp = K_sp - CFG['CPS_WING_WIDTH'] * step

        T0 = CFG['ENTRY_DTE'] / 365
        sp0 = greeks(S0, K_sp, T0, self.r, iv0, 'PE')
        lp0 = greeks(S0, K_lp, T0, self.r, iv0, 'PE')

        net_prem0 = sp0['price'] - lp0['price']
        if net_prem0 <= 0:
            return

        max_loss_per_lot = (CFG['CPS_WING_WIDTH'] * step - net_prem0) * CFG['LOT_SIZE']
        risk = self.cash * CFG['CAPITAL_RISK_PCT']
        lots = min(max(1, int(risk / max_loss_per_lot)) if max_loss_per_lot > 0 else 1,
                   CFG['MAX_LOTS'])
        lots = dynamic_lots(lots, entry_row)

        sl_mult = dynamic_stop_loss(CFG['CPS_STOP_LOSS_MULT'], entry_row, self._hv_mean)

        self._tid += 1
        trade = {
            'id': self._tid, 'strategy': 'Credit Put Spread',
            'expiry': str(expiry.date()), 'entry_date': str(entry_date.date()),
            'entry_spot': round(S0, 2), 'K_short_put': K_sp, 'K_long_put': K_lp,
            'entry_iv': round(iv0, 4), 'entry_net_premium': round(net_prem0, 2),
            'lots': lots, 'exit_date': None, 'exit_reason': None, 'pnl': None,
            'entry_regime': REGIME_NAMES.get(regime, 'UNKNOWN'),
            'entry_regime_conf': round(regime_conf, 3),
        }

        print(f"  ▶  CPS Entry {entry_date.date()} | Exp {expiry.date()} | "
              f"Spot ₹{S0:.0f} | Put [{K_lp}/{K_sp}] | "
              f"Prem ₹{net_prem0:.1f} | Lots {lots} | "
              f"Regime: {REGIME_NAMES.get(regime,'?')} ({regime_conf:.0%})")

        cycle_dates = all_dates[(all_dates >= entry_date) & (all_dates <= expiry)]
        exited = False

        for day_i, date in enumerate(cycle_dates):
            row     = self.df.loc[date]
            S       = float(row['Close'])
            iv_now  = float(row['IV'])
            dte_rem = len(cycle_dates) - day_i
            T       = max(dte_rem / 365, 1e-6)

            sp_now = greeks(S, K_sp, T, self.r, iv_now, 'PE')
            lp_now = greeks(S, K_lp, T, self.r, iv_now, 'PE')

            curr_prem = sp_now['price'] - lp_now['price']
            pnl = (net_prem0 - curr_prem) * lots * CFG['LOT_SIZE']

            profit_target = net_prem0 * lots * CFG['LOT_SIZE'] * CFG['CPS_PROFIT_TARGET']
            sl_debit      = net_prem0 * sl_mult

            # ── Regime-based early exit ───────────────────────────
            curr_regime  = int(row.get('regime', REGIME_UNKNOWN))
            curr_prob_hv = float(row.get('prob_high_vol', 0))

            reason = None
            if   day_i == 0                    : reason = None
            elif pnl >= profit_target          : reason = 'PROFIT_TARGET'
            elif curr_prem >= sl_debit         : reason = 'STOP_LOSS'
            elif dte_rem <= CFG['CPS_MIN_DTE'] : reason = 'TIME_EXIT'
            elif date == expiry                : reason = 'EXPIRY'
            elif (curr_regime == REGIME_HIGH_VOL and
                  curr_prob_hv > 0.55 and day_i > 2) : reason = 'REGIME_EXIT'

            if reason:
                self.cash += pnl
                trade.update({
                    'exit_date': str(date.date()), 'exit_spot': round(S, 2),
                    'exit_reason': reason, 'pnl': round(pnl, 2), 'dte_held': day_i,
                    'exit_iv': round(iv_now, 4),
                })
                icon = '✅' if pnl >= 0 else '❌'
                print(f"     {icon} CPS [{reason:<16}] P&L ₹{pnl:>+8,.0f} | DTE:{day_i}")
                exited = True
                break

        if not exited:
            trade['exit_reason'] = 'OPEN_AT_END'
            trade['pnl'] = 0
        self.trades.append(trade)


# ─────────────────────────────────────────────────────────────────────────
# 7C — LONG STRADDLE (regime-enhanced: transition to HIGH_VOL)
# ─────────────────────────────────────────────────────────────────────────

class LongStraddleBacktester:
    """
    Long Straddle — enhanced with HMM regime.
    Enters when probability of transitioning INTO high-vol state is rising.
    """

    def __init__(self, df: pd.DataFrame, cash: float = None):
        self.df     = df
        self.r      = CFG['RISK_FREE_RATE']
        self.cash   = cash if cash is not None else CFG['INITIAL_CAPITAL']
        self.trades = []
        self.alerts = []
        self._tid   = 0

    def run(self, from_date: str, to_date: str):
        expiries = get_monthly_expiry_dates(self.df.index, from_date, to_date)
        print(f"\n📅  LS: {len(expiries)} monthly expiry cycles\n")
        for exp in expiries:
            self._run_cycle(exp)
        print(f"    ✅  Long Straddle — {len(self.trades)} trades")
        return self.trades, self.alerts

    def _run_cycle(self, expiry: pd.Timestamp):
        all_dates  = self.df.index
        pre_expiry = all_dates[all_dates < expiry]
        if len(pre_expiry) < CFG['ENTRY_DTE'] + 5:
            return

        entry_date = pre_expiry[-CFG['ENTRY_DTE']]
        entry_row  = self.df.loc[entry_date]
        S0         = float(entry_row['Close'])
        iv0        = float(entry_row['IV'])
        iv_hv      = float(entry_row['IV_HV_ratio'])
        hv10       = float(entry_row['HV10'])
        hv30       = float(entry_row['HV30'])

        # ── REGIME FILTER ────────────────────────────────────────
        regime      = int(entry_row.get('regime', REGIME_UNKNOWN))
        prob_highv  = float(entry_row.get('prob_high_vol', 0))
        prob_range  = float(entry_row.get('prob_range_bound', 0))

        # Long straddle works when volatility is ABOUT TO expand.
        # Enter when: (a) high-vol probability is rising (transition signal)
        #   OR (b) currently NOT in high-vol but conditions suggest it's coming
        if regime != REGIME_UNKNOWN:
            # Best case: transition INTO high-vol state
            # prob_high_vol is moderate but rising (not already fully priced)
            if prob_highv < CFG['REGIME_LS_HIGHVOL_PROB']:
                # Also check momentum filter as fallback
                if hv10 < hv30 * 0.90:
                    return
            # Already in full high-vol → IV already expensive, skip
            if regime == REGIME_HIGH_VOL and prob_highv > 0.80:
                return

        # ── Existing filter (relaxed) ────────────────────────────
        if iv_hv > CFG['LS_IV_HV_MAX']:
            return
        if hv10 < hv30 * 0.85:
            return

        K  = round(S0 / CFG['STRIKE_ROUND']) * CFG['STRIKE_ROUND']
        T0 = CFG['ENTRY_DTE'] / 365

        ce0 = greeks(S0, K, T0, self.r, iv0, 'CE')
        pe0 = greeks(S0, K, T0, self.r, iv0, 'PE')

        prem0 = ce0['price'] + pe0['price']
        if prem0 <= 0:
            return

        risk = self.cash * CFG['CAPITAL_RISK_PCT']
        lots = min(max(1, int(risk / (prem0 * CFG['LOT_SIZE']))), CFG['MAX_LOTS'])

        # During UNKNOWN regime (training period), limit to 1 lot for safety
        if regime == REGIME_UNKNOWN:
            lots = 1

        self._tid += 1
        trade = {
            'id': self._tid, 'strategy': 'Long Straddle',
            'expiry': str(expiry.date()), 'entry_date': str(entry_date.date()),
            'entry_spot': round(S0, 2), 'strike': K,
            'entry_iv': round(iv0, 4), 'entry_premium': round(prem0, 2),
            'entry_net_premium': round(prem0, 2),
            'lots': lots, 'exit_date': None, 'exit_reason': None, 'pnl': None,
            'entry_regime': REGIME_NAMES.get(regime, 'UNKNOWN'),
            'entry_prob_highvol': round(prob_highv, 3),
        }

        print(f"  ▶  LS  Entry {entry_date.date()} | Exp {expiry.date()} | "
              f"Spot ₹{S0:.0f} | Strike ₹{K} | "
              f"Debit ₹{prem0:.1f} | Lots {lots} | "
              f"P(HighVol)={prob_highv:.0%}")

        cycle_dates = all_dates[(all_dates >= entry_date) & (all_dates <= expiry)]
        exited = False

        for day_i, date in enumerate(cycle_dates):
            row     = self.df.loc[date]
            S       = float(row['Close'])
            iv_now  = float(row['IV'])
            dte_rem = len(cycle_dates) - day_i
            T       = max(dte_rem / 365, 1e-6)

            ce_now = greeks(S, K, T, self.r, iv_now, 'CE')
            pe_now = greeks(S, K, T, self.r, iv_now, 'PE')

            curr_prem = ce_now['price'] + pe_now['price']
            pnl = (curr_prem - prem0) * lots * CFG['LOT_SIZE']

            iv_chg_pct    = (iv_now - iv0) / iv0
            iv_crush_exit = iv_chg_pct < -CFG['LS_IV_CRUSH_EXIT']
            too_long      = day_i >= CFG['LS_MAX_HOLD_DAYS']

            # Leg-lock: if one side is deep ITM
            call_delta = ce_now['delta']
            put_delta  = pe_now['delta']
            leg_lock   = (call_delta > 0.65 or put_delta < -0.65)

            profit_target  =  prem0 * lots * CFG['LOT_SIZE'] * CFG['LS_PROFIT_TARGET']
            max_loss_limit = -prem0 * lots * CFG['LOT_SIZE'] * CFG['LS_STOP_LOSS_PCT']

            reason = None
            if   day_i == 0                    : reason = None
            elif pnl >= profit_target          : reason = 'PROFIT_TARGET'
            elif leg_lock and pnl > 0          : reason = 'LEG_LOCK_PROFIT'
            elif pnl <= max_loss_limit         : reason = 'STOP_LOSS'
            elif iv_crush_exit                 : reason = 'IV_CRUSH_EXIT'
            elif too_long                      : reason = 'MAX_HOLD_EXIT'
            elif dte_rem <= CFG['LS_MIN_DTE']  : reason = 'TIME_EXIT'
            elif date == expiry                : reason = 'EXPIRY'

            if reason:
                self.cash += pnl
                trade.update({
                    'exit_date': str(date.date()), 'exit_spot': round(S, 2),
                    'exit_reason': reason, 'pnl': round(pnl, 2), 'dte_held': day_i,
                    'exit_iv': round(iv_now, 4),
                })
                icon = '✅' if pnl >= 0 else '❌'
                print(f"     {icon} LS  [{reason:<18}] P&L ₹{pnl:>+8,.0f} | DTE:{day_i}")
                exited = True
                break

        if not exited:
            trade['exit_reason'] = 'OPEN_AT_END'
            trade['pnl'] = 0
        self.trades.append(trade)


# ─────────────────────────────────────────────────────────────────────────
# 7D — BULL PUT SPREAD ON DIPS (regime-filtered)
# ─────────────────────────────────────────────────────────────────────────

class BullPutSpreadDipBacktester:
    """
    Bull Put Spread on Dips — mean-reversion after market dips.
    Regime filter: skip when HIGH_VOL_STRESS regime probability is too high
    (avoid catching falling knives in genuine crashes).
    """

    def __init__(self, df: pd.DataFrame, cash: float = None):
        self.df     = df
        self.r      = CFG['RISK_FREE_RATE']
        self.cash   = cash if cash is not None else CFG['INITIAL_CAPITAL']
        self.trades = []
        self.alerts = []
        self._tid   = 0
        self._last_entry = None
        self._hv_mean = df['HV20'].mean()

    def run(self, from_date: str, to_date: str):
        print(f"\n📅  BPS-Dip: scanning for dip entries\n")
        all_dates = self.df.index
        fd = pd.Timestamp(from_date)
        td = pd.Timestamp(to_date)
        dates_in_range = all_dates[(all_dates >= fd) & (all_dates <= td)]

        for date in dates_in_range:
            self._check_entry(date)

        print(f"    ✅  Bull Put Spread (Dips) — {len(self.trades)} trades")
        return self.trades, self.alerts

    def _check_entry(self, date: pd.Timestamp):
        row    = self.df.loc[date]
        ret_5d = float(row.get('ret_5d', 0))

        # Check dip condition
        if ret_5d > CFG['BPS_DIP_THRESHOLD']:
            return

        # Cooldown
        if self._last_entry is not None:
            days_since = (date - self._last_entry).days
            if days_since < CFG['BPS_COOLDOWN_DAYS']:
                return

        # ── REGIME FILTER ────────────────────────────────────────
        regime     = int(row.get('regime', REGIME_UNKNOWN))
        prob_highv = float(row.get('prob_high_vol', 0))

        if regime != REGIME_UNKNOWN:
            # Skip entry if we're in confirmed HIGH_VOL stress
            # (the dip is likely part of a larger crash, not a mean-reversion opportunity)
            if prob_highv > CFG['REGIME_BPS_HIGHVOL_SKIP']:
                return

        S0   = float(row['Close'])
        iv0  = float(row['IV'])
        step = CFG['STRIKE_ROUND']
        ATM  = round(S0 / step) * step

        K_sp = ATM - CFG['BPS_PUT_OFFSET'] * step
        K_lp = K_sp - CFG['BPS_WING_WIDTH'] * step

        T0   = CFG['BPS_ENTRY_DTE'] / 365

        sp0 = greeks(S0, K_sp, T0, self.r, iv0, 'PE')
        lp0 = greeks(S0, K_lp, T0, self.r, iv0, 'PE')

        net_prem0 = sp0['price'] - lp0['price']
        if net_prem0 <= 0:
            return

        max_loss_per_lot = (CFG['BPS_WING_WIDTH'] * step - net_prem0) * CFG['LOT_SIZE']
        risk = self.cash * CFG['CAPITAL_RISK_PCT']
        lots = min(max(1, int(risk / max_loss_per_lot)) if max_loss_per_lot > 0 else 1,
                   CFG['MAX_LOTS'])

        # Dynamic sizing: halve lots during regime transitions
        lots = dynamic_lots(lots, row)

        sl_mult = dynamic_stop_loss(CFG['BPS_STOP_LOSS_MULT'], row, self._hv_mean)

        self._tid += 1
        self._last_entry = date

        trade = {
            'id': self._tid, 'strategy': 'Bull Put Spread (Dip)',
            'entry_date': str(date.date()), 'entry_spot': round(S0, 2),
            'K_short_put': K_sp, 'K_long_put': K_lp,
            'entry_iv': round(iv0, 4), 'entry_net_premium': round(net_prem0, 2),
            'lots': lots, 'ret_5d': round(ret_5d * 100, 2),
            'exit_date': None, 'exit_reason': None, 'pnl': None,
            'entry_regime': REGIME_NAMES.get(regime, 'UNKNOWN'),
            'entry_prob_highvol': round(prob_highv, 3),
        }

        print(f"  ▶  BPS Entry {date.date()} | 5d-ret {ret_5d*100:.1f}% | "
              f"Spot ₹{S0:.0f} | Put [{K_lp}/{K_sp}] | "
              f"Prem ₹{net_prem0:.1f} | Lots {lots} | "
              f"P(HighVol)={prob_highv:.0%}")

        # Find exit date range
        all_dates = self.df.index
        entry_idx = all_dates.get_loc(date)
        end_idx   = min(entry_idx + CFG['BPS_ENTRY_DTE'] + 5, len(all_dates) - 1)
        cycle_dates = all_dates[entry_idx:end_idx + 1]

        exited = False
        for day_i, d in enumerate(cycle_dates):
            row_d   = self.df.loc[d]
            S       = float(row_d['Close'])
            iv_now  = float(row_d['IV'])
            dte_rem = max(CFG['BPS_ENTRY_DTE'] - day_i, 0)
            T       = max(dte_rem / 365, 1e-6)

            sp_now = greeks(S, K_sp, T, self.r, iv_now, 'PE')
            lp_now = greeks(S, K_lp, T, self.r, iv_now, 'PE')

            curr_prem = sp_now['price'] - lp_now['price']
            pnl = (net_prem0 - curr_prem) * lots * CFG['LOT_SIZE']

            profit_target = net_prem0 * lots * CFG['LOT_SIZE'] * CFG['BPS_PROFIT_TARGET']
            sl_debit      = net_prem0 * sl_mult

            # ── Regime-based early exit ───────────────────────────
            curr_prob_hv = float(row_d.get('prob_high_vol', 0))
            curr_regime  = int(row_d.get('regime', REGIME_UNKNOWN))

            reason = None
            if   day_i == 0                     : reason = None
            elif pnl >= profit_target           : reason = 'PROFIT_TARGET'
            elif curr_prem >= sl_debit          : reason = 'STOP_LOSS'
            elif dte_rem <= CFG['BPS_MIN_DTE']  : reason = 'TIME_EXIT'
            elif day_i >= CFG['BPS_ENTRY_DTE']  : reason = 'EXPIRY'
            # NEW: exit early if regime shifts to high-vol stress
            elif (curr_regime == REGIME_HIGH_VOL and
                  curr_prob_hv > 0.65 and day_i > 1
                  and pnl < 0)                  : reason = 'REGIME_EXIT'

            if reason:
                self.cash += pnl
                trade.update({
                    'exit_date': str(d.date()), 'exit_spot': round(S, 2),
                    'exit_reason': reason, 'pnl': round(pnl, 2), 'dte_held': day_i,
                    'exit_iv': round(iv_now, 4),
                })
                icon = '✅' if pnl >= 0 else '❌'
                print(f"     {icon} BPS [{reason:<16}] P&L ₹{pnl:>+8,.0f} | DTE:{day_i}")
                exited = True
                break

        if not exited:
            trade['exit_reason'] = 'OPEN_AT_END'
            trade['pnl'] = 0
        self.trades.append(trade)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(trades, initial_capital):
    closed = [t for t in trades
              if t.get('pnl') is not None and t.get('exit_reason') != 'OPEN_AT_END']
    if not closed:
        return {}

    pnls    = [t['pnl'] for t in closed]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    cum_pnl = np.cumsum(pnls)
    peak    = np.maximum.accumulate(cum_pnl)
    dd_abs  = cum_pnl - peak
    dd_pct  = dd_abs / (initial_capital + peak) * 100

    exit_counts = {}
    for t in closed:
        r = t.get('exit_reason', 'UNKNOWN')
        exit_counts[r] = exit_counts.get(r, 0) + 1

    strat_pnl = {}
    for t in closed:
        s = t.get('strategy', 'Unknown')
        strat_pnl[s] = strat_pnl.get(s, 0) + t['pnl']

    total_pnl = sum(pnls)
    total_return = total_pnl / initial_capital * 100
    years = 3.0
    annualized_return = ((1 + total_pnl / initial_capital) ** (1 / years) - 1) * 100

    return {
        'total_pnl'          : round(total_pnl, 2),
        'total_return_pct'   : round(total_return, 2),
        'annualized_return'  : round(annualized_return, 2),
        'num_trades'         : len(closed),
        'win_rate'           : round(len(wins) / len(pnls) * 100, 1),
        'avg_win'            : round(np.mean(wins), 2)    if wins   else 0,
        'avg_loss'           : round(np.mean(losses), 2)  if losses else 0,
        'best_trade'         : round(max(pnls), 2),
        'worst_trade'        : round(min(pnls), 2),
        'profit_factor'      : round(sum(wins) / abs(sum(losses)), 2)
                               if losses and sum(losses) != 0 else 999,
        'max_drawdown_pct'   : round(dd_pct.min(), 2) if len(dd_pct) > 0 else 0,
        'exit_counts'        : exit_counts,
        'strat_pnl'          : strat_pnl,
    }


def compute_strategy_metrics(trades, initial_capital):
    """Compute metrics per strategy."""
    strat_trades = {}
    for t in trades:
        s = t.get('strategy', 'Unknown')
        strat_trades.setdefault(s, []).append(t)

    results = {}
    for s, st in strat_trades.items():
        m = compute_metrics(st, initial_capital)
        if m:
            results[s] = m
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — PLOTLY DASHBOARD (enhanced with regime overlay)
# ═══════════════════════════════════════════════════════════════════════════

def plot_multi_strategy_dashboard(all_trades, strat_metrics, combined_metrics,
                                   initial_capital, df=None):
    """Multi-strategy comparison dashboard with regime overlay."""
    if not combined_metrics:
        print("⚠️  Nothing to plot.")
        return

    strategies = list(strat_metrics.keys())

    colors = {
        'Iron Condor'           : '#00d4a0',
        'Credit Put Spread'     : '#4a90e2',
        'Long Straddle'         : '#9b59b6',
        'Bull Put Spread (Dip)' : '#1abc9c',
    }

    regime_colors = {
        REGIME_TRENDING : 'rgba(0, 212, 160, 0.15)',   # green tint
        REGIME_RANGE    : 'rgba(74, 144, 226, 0.15)',   # blue tint
        REGIME_HIGH_VOL : 'rgba(231, 76, 60, 0.15)',    # red tint
    }

    n_rows = 3 if df is not None else 2

    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=(
            '📈 Cumulative P&L by Strategy',
            '📊 Per-Trade P&L (All Strategies)',
            '🥧 Strategy P&L Contribution',
            '🏆 Win Rate by Strategy',
            *(['🧠 NIFTY Price with Regime Overlay',
               '📉 Regime Confidence Over Time'] if df is not None else []),
        ),
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'pie'}, {'type': 'xy'}],
            *([[{'type': 'xy'}, {'type': 'xy'}]] if df is not None else []),
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # ── 1. Cumulative P&L per strategy ─────────────────────────────
    for strat in strategies:
        strat_closed = [t for t in all_trades
                        if t.get('strategy') == strat
                        and t.get('pnl') is not None
                        and t.get('exit_reason') != 'OPEN_AT_END']
        if not strat_closed:
            continue
        strat_closed.sort(key=lambda x: x.get('exit_date', ''))
        cum_pnl = np.cumsum([t['pnl'] for t in strat_closed])
        dates   = [t.get('exit_date', '') for t in strat_closed]
        color   = colors.get(strat, '#ffffff')
        fig.add_trace(go.Scatter(
            x=dates, y=cum_pnl.tolist(), mode='lines+markers',
            name=strat, line=dict(color=color, width=2),
            marker=dict(size=5),
        ), row=1, col=1)

    fig.add_shape(type='line', xref='paper', yref='y1',
                  x0=0, x1=1, y0=0, y1=0,
                  line=dict(dash='dash', color='rgba(255,255,255,0.3)', width=1))

    # ── 2. Per-trade P&L bars ──────────────────────────────────────
    all_closed = [t for t in all_trades
                  if t.get('pnl') is not None and t.get('exit_reason') != 'OPEN_AT_END']
    all_closed.sort(key=lambda x: x.get('exit_date', ''))
    pnls    = [t['pnl'] for t in all_closed]
    t_dates = [t.get('exit_date', '') for t in all_closed]
    bar_col = [colors.get(t.get('strategy', ''), '#ffffff') for t in all_closed]

    fig.add_trace(go.Bar(
        x=t_dates, y=pnls, name='Trade P&L',
        marker_color=bar_col,
        text=[f'₹{p:+,.0f}' for p in pnls],
        textposition='outside', textfont=dict(size=7),
    ), row=1, col=2)

    # ── 3. Strategy P&L pie ────────────────────────────────────────
    strat_pnl = combined_metrics.get('strat_pnl', {})
    if strat_pnl:
        labels = list(strat_pnl.keys())
        values = [max(v, 0) for v in strat_pnl.values()]
        pie_colors = [colors.get(s, '#ffffff') for s in labels]
        fig.add_trace(go.Pie(
            labels=labels, values=values, hole=0.42,
            textinfo='label+percent', textfont=dict(size=9),
            marker=dict(colors=pie_colors),
        ), row=2, col=1)

    # ── 4. Win rate bars ───────────────────────────────────────────
    wr_strats = []
    wr_vals   = []
    wr_colors = []
    for s, m in strat_metrics.items():
        wr_strats.append(s)
        wr_vals.append(m.get('win_rate', 0))
        wr_colors.append(colors.get(s, '#ffffff'))

    fig.add_trace(go.Bar(
        x=wr_strats, y=wr_vals, name='Win Rate %',
        marker_color=wr_colors,
        text=[f'{v:.0f}%' for v in wr_vals],
        textposition='outside', textfont=dict(size=10),
    ), row=2, col=2)

    fig.add_shape(type='line', xref='paper', yref='y4',
                  x0=0, x1=1, y0=50, y1=50,
                  line=dict(dash='dot', color='rgba(255,69,96,0.5)', width=1))

    # ── 5 & 6. Regime overlay (if DataFrame available) ─────────────
    if df is not None:
        valid = df[df['regime'] != REGIME_UNKNOWN].copy()
        if len(valid) > 0:
            # 5. NIFTY price with regime background
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'], mode='lines',
                name='NIFTY Close', line=dict(color='#e2e8f0', width=1.5),
                showlegend=False,
            ), row=3, col=1)

            # Add regime background colors
            regime_trace_names = {
                REGIME_TRENDING: 'Trending', REGIME_RANGE: 'Range-Bound',
                REGIME_HIGH_VOL: 'High-Vol Stress'
            }
            for r_val, r_color in regime_colors.items():
                mask = valid['regime'] == r_val
                if mask.sum() == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=valid.index[mask], y=valid['Close'][mask],
                    mode='markers', name=regime_trace_names.get(r_val, '?'),
                    marker=dict(color=r_color.replace('0.15', '0.7'), size=3),
                    showlegend=True,
                ), row=3, col=1)

            # 6. Regime confidence and probabilities
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid['prob_trending'],
                mode='lines', name='P(Trending)',
                line=dict(color='#00d4a0', width=1),
            ), row=3, col=2)
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid['prob_range_bound'],
                mode='lines', name='P(Range)',
                line=dict(color='#4a90e2', width=1),
            ), row=3, col=2)
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid['prob_high_vol'],
                mode='lines', name='P(HighVol)',
                line=dict(color='#e74c3c', width=1),
            ), row=3, col=2)

    # ── Stats subtitle ────────────────────────────────────────────
    ann_ret = combined_metrics.get('annualized_return', 0)
    stats_text = (
        f"Total Trades: {combined_metrics['num_trades']}  |  "
        f"Win Rate: {combined_metrics['win_rate']}%  |  "
        f"P&L: ₹{combined_metrics['total_pnl']:+,.0f}  |  "
        f"Return: {combined_metrics['total_return_pct']:+.1f}%  |  "
        f"Ann.: {ann_ret:+.1f}%  |  "
        f"PF: {combined_metrics['profit_factor']}  |  "
        f"MaxDD: {combined_metrics['max_drawdown_pct']}%"
    )

    fig.update_layout(
        title=dict(
            text='<b>NIFTY Multi-Strategy Greek-Based Backtester v3.0 — HMM/GMM Regime</b>'
                 f'<br><sup style="color:#94a3b8">{stats_text}</sup>',
            font=dict(size=15, color='#e2e8f0'),
            x=0.5, xanchor='center',
        ),
        paper_bgcolor='#0f172a',
        plot_bgcolor ='#1e293b',
        font=dict(color='#cbd5e1',
                  family='JetBrains Mono, Courier New, monospace'),
        height=500 * n_rows,
        showlegend=True,
        legend=dict(
            x=0.01, y=0.99, bgcolor='rgba(30,41,59,0.8)',
            font=dict(size=9),
        ),
        margin=dict(t=110, b=40, l=55, r=30),
    )

    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=11, color='#94a3b8')

    axis_style = dict(gridcolor='#334155', zeroline=False,
                      tickfont=dict(size=9), showgrid=True)
    for r in range(1, n_rows + 1):
        for c in [1, 2]:
            try:
                fig.update_xaxes(axis_style, row=r, col=c)
                fig.update_yaxes(axis_style, row=r, col=c)
            except Exception:
                pass

    fig.show()
    out_path = 'multi_strategy_dashboard.html'
    fig.write_html(out_path)
    print(f"\n💾  Dashboard saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — PRINT REPORTS
# ═══════════════════════════════════════════════════════════════════════════

def print_combined_summary(combined_metrics, strat_metrics, initial_capital):
    sep = '═' * 70
    print(f"\n{sep}")
    print("  MULTI-STRATEGY BACKTEST SUMMARY v3.0 (HMM/GMM Regime)")
    print(sep)
    ann_ret = combined_metrics.get('annualized_return', 0)
    rows = [
        ('Initial Capital',  f"₹{initial_capital:>12,.0f}"),
        ('Total P&L',        f"₹{combined_metrics['total_pnl']:>+12,.0f}"),
        ('Total Return',     f"{combined_metrics['total_return_pct']:>+11.2f}%"),
        ('Annualized Return', f"{ann_ret:>+11.2f}%"),
        ('No. of Trades',    f"{combined_metrics['num_trades']:>13}"),
        ('Win Rate',         f"{combined_metrics['win_rate']:>12.1f}%"),
        ('Avg Win',          f"₹{combined_metrics['avg_win']:>+12,.0f}"),
        ('Avg Loss',         f"₹{combined_metrics['avg_loss']:>+12,.0f}"),
        ('Best Trade',       f"₹{combined_metrics['best_trade']:>+12,.0f}"),
        ('Worst Trade',      f"₹{combined_metrics['worst_trade']:>+12,.0f}"),
        ('Profit Factor',    f"{combined_metrics['profit_factor']:>13.2f}"),
        ('Max Drawdown',     f"{combined_metrics['max_drawdown_pct']:>12.2f}%"),
    ]
    for label, val in rows:
        print(f"  {label:<20}: {val}")

    print('─' * 70)
    print("  Exit Breakdown:")
    for r, c in combined_metrics.get('exit_counts', {}).items():
        print(f"    {r:<28}: {c}")

    print(f"\n{'─' * 70}")
    print("  P&L BY STRATEGY:")
    print(f"  {'Strategy':<28} {'P&L':>12} {'Trades':>8} {'Win%':>8} {'PF':>8}")
    print(f"  {'─'*28} {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    for s, m in strat_metrics.items():
        pnl_str = f"₹{m['total_pnl']:>+10,.0f}"
        pf_str  = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 900 else "∞"
        print(f"  {s:<28} {pnl_str:>12} {m['num_trades']:>8} "
              f"{m['win_rate']:>7.1f}% {pf_str:>8}")
    print(sep)


def print_trade_log(trades):
    closed = [t for t in trades if t.get('exit_reason') != 'OPEN_AT_END'
              and t.get('pnl') is not None]
    closed.sort(key=lambda x: x.get('entry_date', ''))
    w = 140
    print(f"\n{'─'*w}")
    print(f"  {'#':>3}  {'Strategy':<26}  {'Entry':>12}  {'Exit':>12}  "
          f"{'Lots':>4}  {'Prem':>8}  {'P&L (₹)':>10}  "
          f"{'DTE':>5}  {'Regime':<20}  Exit Reason")
    print(f"{'─'*w}")
    for t in closed:
        regime_str = t.get('entry_regime', '?')
        print(
            f"  {t['id']:>3}  "
            f"{t.get('strategy','?'):<26}  "
            f"{t['entry_date']:>12}  "
            f"{str(t.get('exit_date','—')):>12}  "
            f"{t['lots']:>4}  "
            f"{t.get('entry_net_premium', 0):>8.1f}  "
            f"{t.get('pnl', 0):>+10,.0f}  "
            f"{t.get('dte_held', 0):>5}  "
            f"{regime_str:<20}  "
            f"{t.get('exit_reason','?')}"
        )
    print(f"{'─'*w}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_all_strategies(from_date=None, to_date=None):
    """
    Run all 4 strategies with HMM/GMM regime detection.
    Walk-forward training ensures no look-ahead bias.
    """
    from_date = from_date or CFG['FROM_DATE']
    to_date   = to_date   or CFG['TO_DATE']

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NIFTY Multi-Strategy Greek-Based Backtester v3.0          ║")
    print("║  HMM + GMM Regime Detection | Walk-Forward | BS Greeks    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df = fetch_nifty(from_date, to_date)

    all_trades = []
    all_alerts = []

    # ── Strategy 1: Iron Condor (range-bound regime only) ─────────
    print("\n" + "─"*60)
    print("  Strategy 1: IRON CONDOR (regime-gated → RANGE_BOUND)")
    print("─"*60)
    ic_bt = IronCondorBacktester(df)
    ic_trades, ic_alerts = ic_bt.run(from_date, to_date)
    all_trades.extend(ic_trades)
    all_alerts.extend(ic_alerts)

    # ── Strategy 2: Credit Put Spread (trending regime only) ──────
    print("\n" + "─"*60)
    print("  Strategy 2: CREDIT PUT SPREAD (regime-gated → TRENDING)")
    print("─"*60)
    cps_bt = CreditPutSpreadBacktester(df)
    cps_trades, cps_alerts = cps_bt.run(from_date, to_date)
    all_trades.extend(cps_trades)
    all_alerts.extend(cps_alerts)

    # ── Strategy 3: Long Straddle (high-vol transition) ───────────
    print("\n" + "─"*60)
    print("  Strategy 3: LONG STRADDLE (regime-enhanced → HIGH_VOL transition)")
    print("─"*60)
    ls_bt = LongStraddleBacktester(df)
    ls_trades, ls_alerts = ls_bt.run(from_date, to_date)
    all_trades.extend(ls_trades)
    all_alerts.extend(ls_alerts)

    # ── Strategy 4: Bull Put Spread on Dips (regime-filtered) ─────
    print("\n" + "─"*60)
    print("  Strategy 4: BULL PUT SPREAD ON DIPS ⭐ (regime-filtered)")
    print("─"*60)
    bps_bt = BullPutSpreadDipBacktester(df)
    bps_trades, bps_alerts = bps_bt.run(from_date, to_date)
    all_trades.extend(bps_trades)
    all_alerts.extend(bps_alerts)

    # ── Combined metrics ──────────────────────────────────────────
    combined_metrics = compute_metrics(all_trades, CFG['INITIAL_CAPITAL'])
    strat_metrics    = compute_strategy_metrics(all_trades, CFG['INITIAL_CAPITAL'])

    if not combined_metrics:
        print("⚠️  No closed trades found.")
        return all_trades, all_alerts, combined_metrics, strat_metrics

    # ── Reports ───────────────────────────────────────────────────
    print_combined_summary(combined_metrics, strat_metrics, CFG['INITIAL_CAPITAL'])
    print("\n📋  TRADE LOG")
    print_trade_log(all_trades)

    # ── Dashboard ─────────────────────────────────────────────────
    plot_multi_strategy_dashboard(all_trades, strat_metrics,
                                  combined_metrics, CFG['INITIAL_CAPITAL'],
                                  df=df)

    # ── Save results ──────────────────────────────────────────────
    with open('multi_strategy_results.json', 'w') as f:
        json.dump({
            'combined_metrics': combined_metrics,
            'strategy_metrics': {k: v for k, v in strat_metrics.items()},
            'trades': all_trades,
        }, f, indent=2, default=str)
    print("\n💾  Results saved → multi_strategy_results.json")

    return all_trades, all_alerts, combined_metrics, strat_metrics


# ── Auto-run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_all_strategies()
