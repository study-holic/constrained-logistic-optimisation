# Constrained Logistic Fitting as Optimisation

A framework for fitting a logistic growth curve as an explicit constrained
optimisation problem, comparing three solvers under one shared protocol with
closed-form gradients, a finite-difference Hessian, and convergence diagnostics.

Companion code for the paper:

Constrained Logistic Fitting as Optimisation: A Quantitative Convergence Study. William Odumosu, (2026).

## What this does

Nonlinear curve fitting is usually reported as a single set of fitted
parameters, as if the optimiser were an implementation detail. It is not: the
same model on the same data can land in different places depending on the
method, the constraint handling, and how noise meets curvature. This repo makes
that dependence explicit.

Given noisy observations of a logistic curve, it:

- Poses fitting as constrained least squares with positivity constraints on the growth rate and saturation level
- Derives and implements the closed-form gradient, plus a finite-difference Hessian for the second-order method
- Runs projected gradient descent, projected stochastic gradient descent, and a damped Newton-type method under one shared protocol
- Reports per-iteration loss trajectories and a final-loss comparison
- Identifies where the growth-rate parameter becomes hard to pin down from the data alone

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run the study

```bash
python reproduce.py
```

Generates the synthetic problem, runs all three methods under identical
constraints and iteration budget, and prints the final loss and recovered
parameters for each.

### Save the convergence plot

```bash
python reproduce.py --plot
```

Also writes a log-scale loss-trajectory plot to `figures/convergence.png`.

### Change the seed

```bash
python reproduce.py --seed 7
```

Draws a different noise realisation and mini-batch sequence, so you can check
the behaviour is not an artefact of one run.

## Example output

```
$ python reproduce.py

Method              Final loss   (r, K)
Projected GD        0.09560     (1.558, 9.831)
Projected SGD       0.09543     (1.572, 9.841)
Damped Newton-type  0.08178     (1.484, 10.058)
```

Ground truth is (r, K) = (1.5, 10) with noise std 0.3, so a loss near 0.09
(about sigma squared) is the noise floor: every method recovers the curve, and
the Newton-type method gets there fastest.

## How it works

**Constrained objective:** Fitting minimises the mean-squared error between the
logistic curve and the data, subject to r and K staying positive. The
constraint is enforced by projecting onto the feasible set after each step.

**Gradient and curvature:** The gradient is derived in closed form. Its key
term is s(1-s), the sigmoid slope, which peaks at the inflection point and
vanishes in the saturation tails. When the data barely sample the transition
region the loss becomes weakly sensitive to r, which is an identifiability
problem rather than a solver problem.

**Methods:** Projected GD is the stable, predictable baseline. Projected SGD
moves quickly early on but carries gradient noise. The damped Newton-type
method uses curvature for fast local convergence, with regularisation and a
backtracking line search to keep the step well-behaved when curvature is poorly
conditioned.

**A note on numbers:** The paper reports one representative run. This code uses
NumPy's recommended `default_rng` with independent seeds for the data and the
stochastic optimiser, so exact final losses depend on the seed and NumPy
version. The qualitative picture is stable across seeds.

## File structure

```
model.py          The maths: curve, MSE loss, closed-form gradient, finite-difference Hessian
optimisers.py     The three constrained methods and the shared Config
experiment.py     Data generation and the run loop
reproduce.py      Command-line entry point
requirements.txt  Python dependencies
```

## Citation

```
@misc{odumosu2026logistic,
  author = {Odumosu, William},
  title  = {Constrained Logistic Fitting as Optimisation: A Quantitative Convergence Study},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
