# Bounded setup-optimization pilot

## Observed result

The normalised 10-step objective changed from 4 to 3.88122 (2.970%). The final stiffnesses were k_f=20847.7 N/m and k_r=22371.7 N/m.

## Numerical validation

All production simulations and AD gradients were finite. Central finite differences were evaluated at the initial point, after step 5, and after step 10. IFT comparison: unavailable in this pilot.

## Interpretation

This bounded deterministic pilot tests whether the conditioned state-based objective admits a finite descent trajectory. It does not establish global optimality, model superiority, or a general setup-optimization result.

## Unsupported conclusions and next steps

No conclusion about IFT agreement, competing optimizers, 200-step behaviour, or repeatability over initial setups is supported. Next measure IFT and FD agreement over additional operating points before extending the horizon.
