#!/usr/bin/env python3
"""Exact 0-1 Bayes-risk calculator for the paper's 3-state/3-outcome examples.

This is a reviewer-facing supplementary demonstration, not a replacement for the
43 preserved research scripts or a general Blackwell-order solver.
"""
from __future__ import annotations
from fractions import Fraction as F
from pathlib import Path
import argparse
import json


def load_fraction(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"Expected nonnegative rational number, received {value!r}")
    return F(str(value))


def parse_matrix(rows, name, n=3):
    if not isinstance(rows, list) or len(rows) != n:
        raise ValueError(f"{name} must contain exactly {n} state rows")
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{name}[{i}] must contain exactly 3 outcome probabilities")
        rr = [load_fraction(v) for v in row]
        if any(v < 0 or v > 1 for v in rr) or sum(rr) != 1:
            raise ValueError(f"{name}[{i}] must be a probability row summing to 1")
        out.append(rr)
    return out


def risk01(matrix, prior):
    # Minimum 0-1 Bayes classification risk given known state-conditional experiment.
    return 1 - sum(max(prior[i] * matrix[i][j] for i in range(len(prior))) for j in range(3))


def run(data):
    if not isinstance(data, dict) or not isinstance(data.get("experiments"), dict):
        raise ValueError("Expected a JSON object with 'experiments' mapping")
    prior = [load_fraction(p) for p in data.get('prior', ['1/3'] * 3)]
    if len(prior) != 3 or any(p < 0 for p in prior) or sum(prior) != 1:
        raise ValueError('prior must contain 3 nonnegative probabilities summing to 1')
    result = {'method': 'exact_0-1_Bayes_risk_for_3_states_and_3_observations', 'prior': [str(p) for p in prior], 'experiments': {}}
    for name, pair in sorted(data['experiments'].items()):
        p = parse_matrix(pair['P'], name+'.P')
        q = parse_matrix(pair['Q'], name+'.Q')
        rp, rq = risk01(p, prior), risk01(q, prior)
        result['experiments'][name] = {'P_risk': str(rp), 'Q_risk': str(rq), 'Q_minus_P_risk': str(rq - rp)}
    if 'A' in result['experiments'] and 'B' in result['experiments']:
        da = F(result['experiments']['A']['Q_minus_P_risk'])
        db = F(result['experiments']['B']['Q_minus_P_risk'])
        result['A_minus_B_risk_change'] = str(da - db)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='JSON containing 3-state/3-outcome matrices')
    parser.add_argument('--output',help='Output JSON file; prints to stdout even when omitted')
    args=parser.parse_args()
    result=run(json.loads(Path(args.input).read_text(encoding='utf-8')))
    payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    print(payload,end='')
    if args.output:
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(payload,encoding='utf-8')

if __name__=='__main__':
    main()
