# External reference data (for the Yahoo Finance cross-check)

`tmx/` holds daily XEQT price history downloaded by hand from TMX Money
(money.tmx.com, the Toronto Stock Exchange's market data site) on 2026-11-25,
Trade History > Price History > Download. TMX limits each download to 90 days,
so three windows were sampled: early 2022, early 2025 (validation year) and
August to November 2026 (test year). Files are unmodified apart from the name.

TMX prices are actual traded prices. Yahoo prices in this project are
dividend-adjusted, so the two series differ by a constant factor that steps on
each ex-distribution date.

`xeqt_distributions.csv` lists BlackRock Canada's announced XEQT cash
distributions for those windows, with the press release used for each row.
BlackRock's releases give the record date. The ex-date is the record date
since Canada moved to T+1 settlement in May 2024, and one trading day earlier
before that (T+2).
