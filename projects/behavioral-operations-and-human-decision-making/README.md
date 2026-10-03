# Behavioral Operations and Human Decision-Making

A behavioral Operations / Management Science module built around the canonical newsvendor problem.

It separates the normative benchmark from human decision patterns:

- critical-fractile newsvendor recommendation under Normal demand;
- pull-to-center bias;
- anchoring and insufficient adjustment;
- bounded human overrides around an analytical recommendation;
- Monte Carlo measurement of profit regret created by behavioral deviations.

The override helper is deliberately a governance mechanism rather than a claim that humans or algorithms should always dominate. It makes the permitted intervention range explicit and testable.

Run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

The code is intended for controlled behavioral experiments and teaching. Real human-in-the-loop systems require preregistered evaluation, interface testing, heterogeneous-user analysis, and organizational governance.
