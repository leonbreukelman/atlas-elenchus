# Architecture

## Vanilla Pipeline

```
Market Data → [L1: Macro Agents] → [L2: Sector Agents] → [L3: Decision Agents] → Portfolio
                                                                                    ↓
                                                                              Sharpe Ratio
                                                                                    ↓
                                                                        Darwinian Weight Update
                                                                                    ↓
                                                                   Autoresearch (mutate worst agent)
```

## Elenchus Pipeline

```
Market Data → [L1: Macro Agents] → Elenchus Probe → [L2: Sector Agents] → Elenchus Probe → [L3: Decision] → Elenchus Probe → Portfolio
                                        ↓                                       ↓                                  ↓
                                   deutsch_score                           deutsch_score                      deutsch_score
                                        ↓                                       ↓                                  ↓
                                  Filter easy-to-vary                    Filter easy-to-vary               Weight by deutsch_score
                                                                                                                   ↓
                                                                                                             Sharpe Ratio
                                                                                                                   ↓
                                                                                                     Darwinian Weight Update
                                                                                                                   ↓
                                                                                                Autoresearch (mutate worst agent)
```

Note: probing is configurable via `--probe-layers`. The production 250-day run used L3-only probing (`--probe-layers 3`), which applies the Deutsch Probe to the CIO and risk_officer outputs only, reducing probe API calls by ~75%.

## Deutsch Probe Detail

```
Recommendation: NVDA long (conviction 0.85)
  Component 1: "Semiconductor capex cycle accelerating per TSMC guidance"
  Component 2: "Relative strength vs SOX over trailing 20d"  
  Component 3: "Data center revenue mix shift reduces cyclicality"
  Component 4: "Risk-on regime favors high-beta tech"

Probe Component 1:
  Replace with: "Semiconductor capex cycle decelerating as hyperscaler orders plateau"
  Does conclusion (NVDA long) still follow from modified reasoning?
  → NO: component was load-bearing ✓

Probe Component 4:
  Replace with: "Risk-off regime favors defensive positioning"
  Does conclusion (NVDA long) still follow from modified reasoning?
  → Depends on remaining components — tests whether this was additive or decorative
```

## Autoresearch Loop

```
Every N trading days (configurable via --mutation-interval, default: 20):
  1. Rank agents by rolling Sharpe
  2. Score quartiles → adjust Darwinian weights
  3. Identify worst agent
  4. Generate targeted prompt mutation via LLM
  5. Apply mutation
  6. Evaluate over next period
  7. git commit (keep) or git checkout (revert)
```
