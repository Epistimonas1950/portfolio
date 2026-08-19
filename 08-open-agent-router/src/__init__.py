"""A cost-optimal router over a fleet of open-weight models.

Every LLM call an agent makes is a decision: which model do you spend on? This
package treats that as a budgeted contextual bandit and the escalation rule as a
split-conformal test, and measures both against a fleet whose ground truth is known.

The organising equation is the per-call reward

    r_t  =  quality_t  -  lambda * cost_t

with cost measured in wall-clock seconds (src/cost.py), quality in whether the call
succeeded, and the arm choice made from a serving-time feature vector (src/features.py).

No model in this package is real. The fleet is simulated (src/fleet/simulator.py) so
that the oracle policy is computable and regret is *measured* rather than estimated;
src/fleet/client.py is the same interface over a real server, and is not runnable here.
Every number this repo reports is a simulated-fleet number.
"""
