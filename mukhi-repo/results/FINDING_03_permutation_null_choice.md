# Finding 03: the permutation null that works for CKA is invalid for the paired statistic

Found by an integration test, 2026-09-20.

## What happened

The paired displacement statistic (FINDING_02) was first shipped with the same
row-permutation baseline used for CKA: shuffle B's rows, recompute, treat the
mean as the floor.

It produced nonsense. In a synthetic model where the script displacement was
planted to FADE from strong to zero with depth, the reported delta came out
near zero at the strong layer and slightly POSITIVE at the zero layer -- the
opposite of the planted truth.

## Why

A consistent script displacement is a **constant offset**:

    delta_i = h_b(i) - h_a(i) ~ c + noise_i

Row-permuting B leaves `c` completely intact, because `c` does not depend on
which sentence is paired with which. The "baseline" therefore contains the
entire signal, and `delta = real - baseline` cancels to zero.

Row permutation is valid for CKA because CKA measures whether the *relational
geometry* of the two sets corresponds item-by-item, and shuffling destroys
exactly that. It is invalid for a statistic about a shared mean direction.

## The fix

Sign-flip each displacement independently:

    delta_i -> s_i * delta_i,   s_i in {-1, +1}

This destroys any shared direction while preserving every magnitude. It agrees
with the analytic floor for isotropic noise, E[consistency] = 1/sqrt(n), which
the pipeline now reports alongside as `analytic_floor` (measured 0.0517 vs
analytic 0.0577 at n=300).

After the fix, on the same planted data:

| model                         | layer 0 | layer 11 |
|-------------------------------|--------:|---------:|
| converging (script fades)     |   0.677 |   -0.001 |
| null (nothing changes)        |   0.665 |    0.671 |

Both now match the planted truth.

## Transferable lesson

Match the null to the statistic's invariances, not to the pipeline's habits.
A permutation scheme is only a valid null if it destroys the specific
structure the statistic measures. Any new statistic added to this repo needs
its null justified on those grounds and tested against planted data.
