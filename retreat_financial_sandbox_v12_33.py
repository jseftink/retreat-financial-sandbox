\
# retreat_financial_sandbox_v12_33.py
# -------------------------------------------------
# Based on v12_26 with additions:
# • Columns: Revenue → Wages → OpEx → NOP → DebtSvc → FCF → W‑2 Multiple → ROA
# • W‑2 Multiple KPI in Audit & table
# • Minor copy tweaks, version bump
# -------------------------------------------------

import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt

APP_VERSION = "12.33"
st.set_page_config(page_title=f"Retreat Financial Sandbox v{APP_VERSION}", layout="wide")

# ----------------------------- helpers -----------------------------
def loan_payment(principal: float, rate_pct: float, years: int) -> float:
    if principal == 0 or rate_pct == 0:
        return 0.0
    r = rate_pct / 100 / 12
    n = years * 12
    return principal * r / (1 - (1 + r) ** -n)

def remaining_balance(principal: float, rate_pct: float, years: int, payments_made_months: int) -> float:
    if principal == 0 or rate_pct == 0:
        return 0.0
    r = rate_pct / 100 / 12
    n = years * 12
    p = payments_made_months
    bal = principal * ((1 + r) ** n - (1 + r) ** p) / ((1 + r) ** n - 1)
    return max(bal, 0.0)

def build_vector(start: float, end: float, periods: int) -> np.ndarray:
    """Linearly scale from start to end over first 5 years, then hold at end."""
    if periods <= 5:
        return np.linspace(start, end, periods)
    first5 = np.linspace(start, end, 5)
    tail = np.full(periods - 5, end)
    return np.concatenate([first5, tail])

# ----------------------------- sidebar -----------------------------
sb = st.sidebar
sb.header("Inputs")

land   = sb.number_input("Land Cost",              0, 1_000_000,  50_000, 1_000)
cabin  = sb.number_input("Cabin Cost",             0, 1_000_000,  70_000, 1_000)
camp   = sb.number_input("Campsite Cost",          0, 1_000_000,  10_000, 1_000)
dome   = sb.number_input("Dome & Other Build‑out", 0, 1_000_000,  70_000, 1_000)

amber_eq = sb.number_input("Amber Equity",         0, 1_000_000,  30_000, 1_000)
jason_eq = sb.number_input("Jason Equity",         0, 1_000_000,  30_000, 1_000)

owner_loan = sb.number_input("0 % Owner Loan",     0, 1_000_000, 100_000, 1_000)
rate       = sb.number_input("Bank Loan Rate %",   0.0, 20.0, 7.0, 0.1)
term       = sb.number_input("Bank Loan Term (yrs)", 1, 30, 15)

total_assets = land + cabin + camp + dome
conv_gap = max(0, total_assets - amber_eq - jason_eq - owner_loan)
sb.info(f"Conventional financed: ${conv_gap:,.0f}")

sb.markdown("### Package Prices (2‑night stays)")
price_camp  = sb.slider("Campsite + Dome price",            100, 1000, 350, 25)
price_cabin = sb.slider("Cabin + Dome price",               200, 1500, 650, 25)
price_prem  = sb.slider("Cabin Premium price",              300, 2000, 800, 25)
price_day   = sb.slider("Day‑Pass price",                    50,  600, 175, 5)

sb.markdown("### Mix % of Nights")
mix_camp  = sb.slider("Campsite share %", 0, 100, 30)
mix_cabin = sb.slider("Cabin share %",    0, 100 - mix_camp, 40)
mix_prem  = 100 - mix_camp - mix_cabin
sb.text(f"Cabin Premium share = {mix_prem}%")

sb.markdown("### Occupancy & Growth")
occ_start    = sb.slider("Year‑1 Occupancy %", 0, 100, 20)
occ_end      = sb.slider("Year‑5 Occupancy %", 0, 100, 45)
day_passes   = sb.number_input("Day‑Pass Sessions per yr", 0, 3000, 120, 10)
price_growth = sb.slider("Real Price Growth %", 0.0, 10.0, 0.0, 0.1)

sb.markdown("### Operating Costs")
wage_session = sb.number_input("Loaded wage per 90‑min session ($)", 0.0, 500.0, 52.5, 0.5)
cola         = sb.slider("Wage COLA %", 0.0, 10.0, 4.0, 0.1)
maint_pct    = sb.slider("Maintenance % of Structures", 0.0, 10.0, 1.5, 0.1)
admin_pct    = sb.slider("Admin % of Revenue",          0.0, 20.0, 8.0, 0.5)
fixed_opex   = sb.number_input("Other Fixed OPEX per yr",    0, 100_000, 5_000, 500)

sb.markdown("### Exit")
exit_mult = sb.slider("NOP Multiple (x)", 2, 6, 3)
land_app  = sb.slider("Land/Improv. Appreciation %", 0.0, 10.0, 2.0, 0.1)
sale_year = sb.slider("Sale Year", 5, 30, 10)

# ----------------------------- model -----------------------------
nights_cap = 365
occ_vec = build_vector(occ_start/100, occ_end/100, sale_year)
mix_vec = np.array([mix_camp, mix_cabin, mix_prem]) / 100
price_vec = np.array([price_camp, price_cabin, price_prem])
bank_pmt_month = loan_payment(conv_gap, rate, term)

rows = []
for yr in range(1, sale_year + 1):
    occ = occ_vec[yr - 1]
    nights = occ * nights_cap * mix_vec
    stays = nights / 2
    pf = (1 + price_growth / 100) ** (yr - 1)

    rev_components = stays * price_vec * pf
    rev_day = day_passes * price_day * pf
    revenue = rev_components.sum() + rev_day

    sessions = stays[0] + stays[1] + stays[2] * 2 + day_passes
    wage = wage_session * ((1 + cola / 100) ** (yr - 1)) * sessions
    maintenance = (cabin + camp + dome) * maint_pct / 100
    admin = revenue * admin_pct / 100
    opex = maintenance + admin + fixed_opex
    nop = revenue - wage - opex

    debt = bank_pmt_month * 12 if yr <= term else 0
    fcf = nop - debt
    roa = nop / total_assets if total_assets else 0
    w2_multiple = (revenue - opex) / wage if wage else 0

    rows.append(dict(
        Year=yr,
        OccRate=occ*100,
        CampStays=stays[0],
        CabinStays=stays[1],
        PremStays=stays[2],
        DayPasses=day_passes,
        Revenue=revenue,
        Wages=wage,
        OpEx=opex,
        NOP=nop,
        DebtSvc=debt,
        FCF=fcf,
        W2Multiple=w2_multiple,
        ROA=roa
    ))

df = pd.DataFrame(rows)

# ----------------------------- exit & equity proceeds -----------------------------
land_exit = land * ((1 + land_app / 100) ** sale_year)
improve_exit = (cabin + camp + dome) * ((1 + land_app / 100) ** sale_year)
asset_app_exit = land_exit + improve_exit
yearN_nop = df.loc[df.Year == sale_year, "NOP"].values[0]
bank_balance = remaining_balance(conv_gap, rate, term, min(sale_year, term) * 12)

exit_val = asset_app_exit + yearN_nop * exit_mult - bank_balance - owner_loan
cumulative_nop = df["NOP"].sum()
equity_proceeds = cumulative_nop + exit_val

# ----------------------------- IRR & payback -----------------------------
cashflows = df["FCF"].tolist()
cashflows[-1] += exit_val
equity_in = amber_eq + jason_eq
irr = npf.irr([-equity_in] + cashflows)

cum = -equity_in
payback = None
for yr, fc in enumerate(df["FCF"], 1):
    cum += fc
    if cum >= 0:
        payback = yr
        break

# ----------------------------- UI -----------------------------
st.title("Retreat Financial Sandbox")
st.caption(f"Version {APP_VERSION}")

# --- headline metrics ---
c1, c2, c3 = st.columns(3)
c1.metric("Equity Proceeds (Exit + NOP)", f"${equity_proceeds:,.0f}")
c2.metric("Equity IRR", f"{irr*100:.1f}%")
c3.metric("Payback (yrs)", payback or "N/A")

# --- projection table ---
fmt = {"OccRate":"{:.1f}%", "CampStays":"{:,.0f}", "CabinStays":"{:,.0f}",
       "PremStays":"{:,.0f}", "DayPasses":"{:,.0f}", "Revenue":"${:,.0f}",
       "Wages":"${:,.0f}", "OpEx":"${:,.0f}", "NOP":"${:,.0f}",
       "DebtSvc":"${:,.0f}", "FCF":"${:,.0f}", "W2Multiple":"{:.2f}×",
       "ROA":"{:.1%}"}

cols_order = ["Year","OccRate","CampStays","CabinStays","PremStays","DayPasses",
              "Revenue","Wages","OpEx","NOP","DebtSvc","FCF","W2Multiple","ROA"]
st.subheader("Annual Projection")
st.dataframe(df[cols_order].style.format(fmt), use_container_width=True)

# --- IRR sensitivity chart ---
st.subheader("IRR vs NOP Multiple")
muls = np.linspace(2, 6, 9)
irr_vals = []
base_cf = df["FCF"].tolist()
for m in muls:
    cf = base_cf[:-1] + [base_cf[-1] + asset_app_exit + yearN_nop * m - bank_balance - owner_loan]
    irr_vals.append(npf.irr([-equity_in] + cf) * 100)

fig, ax = plt.subplots(figsize=(5, 2.5), dpi=110)
ax.plot(muls, irr_vals, marker="o")
ax.axhline(16, linestyle="--")
ax.grid(ls="--", lw=0.4, alpha=0.6)
ax.set_xlabel("NOP Multiple (x)")
ax.set_ylabel("Equity IRR (%)")
st.pyplot(fig, use_container_width=True)

# ----------------------------- Audit section -----------------------------
package_info = """
### Package Definitions
* **Campsite + Dome**   — 2‑night campsite stay and one 90‑minute dome session  
* **Cabin + Dome**      — 2‑night cabin stay and one 90‑minute dome session  
* **Cabin Premium**     — 2‑night cabin stay and two 90‑minute dome sessions  
* **Day‑Pass**          — 3‑hour dome usage and one 90‑minute session
"""

key_formulas = """
### Key Formulas
* **NOP**  = Revenue − Wages − OpEx  
* **FCF**  = NOP − Bank‑loan Debt Service  
* **Exit** = Land & Improv. appreciation + (Final NOP × Multiple) − Bank balance − Owner loan  
* **Equity Proceeds** = Σ NOP + Exit  
* **IRR**  = Internal rate of return on equity cashflows (−Equity + FCFs + Exit)  
* **W‑2 Multiple** = (Revenue − OpEx) ÷ Wages
"""

# --- Year‑1 breakdown ---
y1 = df.iloc[0]
camp_rev_y1  = y1.CampStays * price_camp
cabin_rev_y1 = y1.CabinStays * price_cabin
prem_rev_y1  = y1.PremStays * price_prem
day_rev_y1   = day_passes * price_day
maint1 = (cabin + camp + dome) * maint_pct / 100
admin1 = y1.Revenue * admin_pct / 100
opex1  = maint1 + admin1 + fixed_opex
sessions1 = int(y1.CampStays + y1.CabinStays + y1.PremStays*2 + day_passes)
year1_text = f"""
### Year‑1 Walk‑through
* Occupancy: {y1.OccRate:.1f}%  
* Day‑Pass sessions: {day_passes}

**Revenue**
* Campsite + Dome: {y1.CampStays:.0f} × ${price_camp} = ${camp_rev_y1:,.0f}  
* Cabin + Dome   : {y1.CabinStays:.0f} × ${price_cabin} = ${cabin_rev_y1:,.0f}  
* Cabin Premium  : {y1.PremStays:.0f} × ${price_prem} = ${prem_rev_y1:,.0f}  
* Day‑Pass       : {day_passes} × ${price_day} = ${day_rev_y1:,.0f}  
* **Total Revenue** = ${y1.Revenue:,.0f}

**Operating Costs**
* Wages          : ${y1.Wages:,.0f} (sessions: {sessions1})  
* Maintenance    : ${maint1:,.0f}  
* Admin          : ${admin1:,.0f}  
* Fixed OPEX     : ${fixed_opex:,.0f}  
* **Total OpEx** = ${opex1:,.0f}

**NOP** = ${y1.NOP:,.0f}  
**ROA** = {y1.ROA:.1%}
"""

exit_breakdown = f"""
### Exit Calculation (Year {sale_year})
* Land & improvements appreciation: ${asset_app_exit:,.0f}  
* NOP × multiple: ${yearN_nop:,.0f} × {exit_mult} = ${yearN_nop*exit_mult:,.0f}  
* Less bank‑loan balance: −${bank_balance:,.0f}  
* Less owner‑loan repayment: −${owner_loan:,.0f}  
* **Total Exit Proceeds** = ${exit_val:,.0f}

Σ NOP (to year {sale_year}): ${cumulative_nop:,.0f}  
**Equity Proceeds** = Σ NOP + Exit = ${equity_proceeds:,.0f}
"""

with st.expander("📜 Audit Details"):
    st.markdown(package_info)
    st.markdown(key_formulas)
    st.markdown(exit_breakdown)
    st.markdown(year1_text)

st.caption("© 2025 Micro Retreat • v12.33 — W‑2 multiple formula corrected")
