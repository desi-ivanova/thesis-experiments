from copy import deepcopy
import os
from tqdm import trange, tqdm
from typing import Callable, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ensure the backend is set
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch import Tensor, Size
import torch.distributions as dist
import torch.nn as nn

# import keras
from keras.src.backend.common import global_state

global_state.set_global_attribute("torch_device", "cpu")

# for BayesFlow devs: this ensures that the latest dev version can be found
import sys

sys.path.append("../BayesFlow")
# sys.path.insert(0, "/Users/zizi/Documents/code/sc_mi/BayesFlow")

import bayesflow as bf
from cmdstanpy import CmdStanModel
from joblib import Parallel, delayed

import logging

# disable the long printouts from stan
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


### Then model
# The posterior is symmetric wrt the sources.
# E.g. if we have K=2, then P(\theta_1, \theta_2 | data) = P(\theta_2, \theta_1 | data).
# How can we exploit this symmetry in the inference?


class LocationFinding(nn.Module):
    def __init__(
        self,
        K: int,
        p: int,
        T: int,
        design_func: Callable,
        prior_samples: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.design_func = design_func
        self._prior_samples = prior_samples
        # prior hyperparams
        self.register_buffer("theta_loc", torch.zeros((K, p)))
        self.register_buffer("theta_covmat", torch.eye(p))

        # Observations noise scale:
        self.register_buffer("noise_scale", torch.tensor(0.5))
        self.p = p  # dimension of theta (location finding example will be 1, 2 or 3).
        self.K = K  # number of sources
        self.T = T  # number of experiments

        # fixed hyperparameters
        self.base_signal = 0.1
        self.max_signal = 1e-4
        self._stan_data = {
            "K": self.K,
            # "T": self.T,
            "p": self.p,
            "base_signal": self.base_signal,
            "max_signal": self.max_signal,
            "noise_sd": self.noise_scale.item(),
        }

    def prior(self) -> dist.Distribution:
        if self._prior_samples is None:
            return dist.MultivariateNormal(self.theta_loc, self.theta_covmat)
        else:
            return EmpiricalPrior(self._prior_samples.to(self.theta_loc.device))  # type: ignore

    def outcome_likelihood(self, theta: Tensor, design: Tensor) -> dist.Distribution:
        # inverse proportional to distane between designs and theta
        # design is of shape [batch_shape, p] theta is of shape [batch_shape, K, p]
        sq_dist = (design.unsqueeze(-2) - theta).pow(2).sum(dim=-1)  # [B, K]
        sq_two_norm_inverse = (self.max_signal + sq_dist).pow(-1)  # [B, K]
        # sum over the K sources, add base signal and take log ->  [B, 1]
        mean_y = torch.log(self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True))
        return dist.Normal(mean_y, self.noise_scale)

    def _forward(self, theta: Tensor, batch_shape: Size) -> Tuple[Tensor, Tensor]:
        designs = []
        outcomes = []
        xi_1 = self.design_func(past_designs=None, past_outcomes=None)  # [1, p]
        # expand the first dimension to be batch_shape
        xi_1 = xi_1.expand(batch_shape + xi_1.shape[1:])  # [B, p]

        y_1 = self.outcome_likelihood(theta, xi_1).rsample()  # [B, 1]
        designs.append(xi_1)
        outcomes.append(y_1)

        for _ in range(1, self.T):
            xi_t = self.design_func(
                past_designs=torch.stack(designs, dim=-1),  # [B, p, t]
                past_outcomes=torch.stack(outcomes, dim=-1),  # [B, 1, t]
            )  # [[B], p]
            y_t = self.outcome_likelihood(theta, xi_t).rsample()  # [B, 1]
            designs.append(xi_t)
            outcomes.append(y_t)

        return torch.stack(designs, dim=-1), torch.stack(outcomes, dim=-1)

    def forward(self, batch_shape: Size) -> dict[str, Tensor]:
        theta = self.prior().sample(batch_shape)  # [B, K, p]
        designs, outcomes = self._forward(theta, batch_shape)
        # theta is [B, K, p]; designs are [B, p, T]; outcomes are [B, 1, T]
        return {"params": theta, "designs": designs, "outcomes": outcomes}

    @torch.no_grad()
    def run_policy(self, theta: Tensor) -> dict[str, Tensor]:
        designs, outcomes = self._forward(theta, theta.shape[:-2])
        return {"designs": designs, "outcomes": outcomes}

    def run_hmc_posterior(
        self,
        designs: Tensor,  # [B, p, T]
        outcomes: Tensor,  # [B, 1, T]
        num_samples: int = 1000,
        num_chains: int = 4,
        symmetrise: bool = False,
    ) -> Tensor:
        # outcomes is of shape [B, 1, T]; check:
        assert len(outcomes.shape) == 3 and outcomes.shape[-2] == 1
        B, _, t = outcomes.shape
        stan_model = CmdStanModel(stan_file="location_finding.stan")
        hmc_posterior_samples = []

        for b in range(B):
            data = {
                **self._stan_data,
                "T": t,
                "y": outcomes[b, 0, :].numpy(),  # [T]
                "x": designs[b, ...].numpy().T,  # [p, T] -> [T, p]
            }
            fit = stan_model.sample(
                data=data,
                inits=0.0,
                iter_warmup=num_samples * 5,
                iter_sampling=num_samples,
                chains=num_chains,
                show_progress=False,
                max_treedepth=15,
                adapt_delta=0.99,
            )
            samples = torch.tensor(
                fit.stan_variable("theta"), dtype=torch.float32
            )  # [N, K, p]
            # the posterior is symmetric wrt the sources, so can symmetrise:
            if symmetrise:
                total_samples = num_samples * num_chains
                samples = samples.reshape(total_samples * self.K, self.p)  # [K*N, p]
                samples = samples.unsqueeze(-2).expand(
                    total_samples * self.K, self.K, self.p
                )  # [K*N, K, p]

            hmc_posterior_samples.append(samples)
        return torch.stack(hmc_posterior_samples, dim=0)  # [B, N, K, p]

    def plot_realisations(
        self, params: Tensor | None, designs: Tensor | None, outcomes: Tensor | None
    ) -> None:
        if params is None:
            sample_dict = self(batch_shape=torch.Size([1]))
            params = sample_dict["params"]
            designs = sample_dict["designs"]
            outcomes = sample_dict["outcomes"]
            B = 1
        else:
            *batch_shape, K, p = params.shape
            assert len(batch_shape) == 1 and p == 2
            B = batch_shape[0]

        assert designs is not None and outcomes is not None and params is not None

        # plot the signal strength on a 2D grid
        grid_size = 100
        _grid_x, _grid_y = torch.meshgrid(
            torch.linspace(-4, 4, grid_size), torch.linspace(-4, 4, grid_size)
        )
        _designs = torch.stack([_grid_x, _grid_y], dim=-1)  # [g, g, 2]

        # Create a grid of subplots
        fig, axes = plt.subplots(
            nrows=1,
            ncols=B,
            figsize=(4 * B, 4),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        # Initialize variables to store min and max values
        min_signal, max_signal = float("inf"), float("-inf")
        min_outcome, max_outcome = float("inf"), float("-inf")

        # First pass to determine global min and max values
        for b in range(B):
            mean_signal = self.outcome_likelihood(
                params[b, ...], _designs
            ).mean  # [g, g, 1]
            min_signal = min(min_signal, mean_signal.min().item())
            max_signal = max(max_signal, mean_signal.max().item())
            min_outcome = min(min_outcome, outcomes[b, 0, ...].min().item())
            max_outcome = max(max_outcome, outcomes[b, 0, ...].max().item())

        # Second pass to plot with consistent scales
        for b in range(B):
            ax = axes[b]

            test_theta = params[b, ...]  # [K, p]
            test_designs = designs[b, ...]  # [p, T]
            test_outcomes = outcomes[b, 0, ...]  # [T]

            mean_signal = self.outcome_likelihood(
                test_theta, _designs
            ).mean  # [g, g, 1]
            contour = ax.contourf(
                _grid_x,
                _grid_y,
                mean_signal.squeeze(-1),
                levels=50,
                cmap="viridis",
                alpha=0.9,
                vmin=min_signal,
                vmax=max_signal,
            )
            scatter = ax.scatter(
                test_designs[0, :],
                test_designs[1, :],
                c=test_outcomes,
                cmap="Reds",
                vmin=min_outcome,
                vmax=max_outcome,
            )
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.scatter(test_theta[:, 0], test_theta[:, 1], c="red", s=20, marker="x")

            ax.set_title(f"Param realisation {b+1}")
            ax.set_xlabel(r"$\xi_1$")
            ax.set_ylabel(r"$\xi_2$")

        # Remove any unused subplots
        for i in range(B, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout to make room for colorbars at the bottom
        fig.subplots_adjust(bottom=0.2, hspace=0.4)

        # Add shared colorbars at the bottom
        cbar_ax1 = fig.add_axes((0.1, 0.08, 0.35, 0.03))
        cbar_ax2 = fig.add_axes((0.55, 0.08, 0.35, 0.03))

        fig.colorbar(
            contour, cax=cbar_ax1, orientation="horizontal", label="Signal Strength"
        )
        fig.colorbar(
            scatter, cax=cbar_ax2, orientation="horizontal", label="Outcome Value"
        )

        plt.show()


def plot_posterior_comparison(
    true_theta: Tensor,  # [B, K, p]
    amortised_samples: Tensor | None = None,  # [B, N_post, K, p]
    hmc_samples: Tensor | None = None,  # [B, N_post, K, p]
    prior_samples: Tensor | None = None,  # [B, K, p]
):
    B, K, p = true_theta.shape
    assert p == 2, "Only 2D plots are supported"
    fig, axs = plt.subplots(
        2, B, figsize=(4 * B, 6), sharex=True, sharey=True, squeeze=False
    )
    fig.suptitle("CouplingFlow NPE vs HMC", fontsize=16, y=1.05)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    plot_settings = {
        "prior": {"color": "gray", "name": "Prior", "marker": "o"},
        "amortised": {"color": "blue", "name": "Amortised Posterior"},
        "hmc": {"color": "green", "name": "HMC Posterior"},
        "true": {"color": "red", "name": "True parameter", "marker": "X"},
    }

    # Generate color gradients
    cmaps = {"params": plt.cm.Reds, "amortised": plt.cm.Blues, "hmc": plt.cm.Greens}  # type: ignore
    colors = {key: cmap(np.linspace(0.3, 0.9, K)[::-1]) for key, cmap in cmaps.items()}

    legend_elements = []

    for row, (samples, sample_type) in enumerate(
        [(amortised_samples, "amortised"), (hmc_samples, "hmc")]
    ):
        for i in range(B):
            ax = axs[row, i]

            if prior_samples is not None:
                ax.scatter(
                    prior_samples[:, :, 0],
                    prior_samples[:, :, 1],
                    alpha=0.2,
                    s=3,
                    color=plot_settings["prior"]["color"],
                    marker=plot_settings["prior"]["marker"],
                )
                if i == 0 and row == 0:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=plot_settings["prior"]["color"],
                            markersize=5,
                            label=plot_settings["prior"]["name"],
                        )
                    )

            if samples is not None:
                for k in range(K):
                    ax.scatter(
                        samples[i, :, k, 0],
                        samples[i, :, k, 1],
                        alpha=0.5,
                        s=3,
                        color=colors[sample_type][k],
                    )
                    if i == 0:
                        legend_elements.append(
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                markerfacecolor=colors[sample_type][k],
                                markersize=5,
                                label=f"{plot_settings[sample_type]['name']} k={k+1}",
                            )
                        )

            for k in range(K):
                ax.scatter(
                    true_theta[i, k, 0],
                    true_theta[i, k, 1],
                    color=colors["params"][k],
                    s=20,
                    marker=plot_settings["true"]["marker"],
                )
                if i == 0 and row == 0:
                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker="X",
                            color="w",
                            markerfacecolor=colors["params"][k],
                            markersize=5,
                            label=f"{plot_settings['true']['name']} k={k+1}",
                        )
                    )

            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_xlabel(r"$\theta_1$")
            ax.set_ylabel(r"$\theta_2$")
            ax.set_title(
                f"{'Amortised' if row == 0 else 'HMC'}: Param realisation {i+1}"
            )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.2, hspace=0.3)

    legend_elements.sort(key=lambda x: x.get_label())
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.02),
        fontsize="small",
    )
    plt.show()


class StaticDesign(nn.Module):
    """
    Fixed (static) design strategy.

    Example:
    B, p, T = 10, 2, 3

    static_design = StaticDesign(designs = torch.randn(T, p), learn_designs = False)
    design = static_design()
    assert design.shape == (1, p)


    design = static_design(past_designs = torch.randn(B, p, T-1))
    assert design.shape == (B, p)
    """

    def __init__(self, designs: Tensor, learn_designs: bool = True) -> None:
        super().__init__()
        if learn_designs:
            self.register_parameter("designs", nn.Parameter(designs))
        else:
            self.register_buffer("designs", designs)

    def forward(
        self, past_designs: Tensor | None = None, past_outcomes: Tensor | None = None
    ) -> Tensor:
        if past_designs is not None:
            *batch_shape, p, t = past_designs.shape  # [B, p, t]
        else:
            batch_shape, t = (1,), 0
        # designs is a tensor of shape [batch_size, p]
        return self.designs[t].unsqueeze(0).expand(*batch_shape, *self.designs[t].shape)


class RandomDesign(nn.Module):
    """
    Random design strategy.

    Example:
    B, p, T = 10, 2, 3

    rd = RandomDesign(design_shape = (p,))
    design = rd()
    assert design.shape == (1, p)

    B, p, T = 10, 2, 3
    design = rd(past_designs = torch.randn(B, p, T))
    assert design.shape == (B, p)
    """

    def __init__(
        self, design_shape: torch.Size, design_dist: dist.Distribution | None = None
    ) -> None:
        super().__init__()
        self.design_shape = design_shape
        self.design_dist = design_dist
        if design_dist is None:
            self.register_buffer("design_mean", torch.zeros(*design_shape))

    def forward(
        self, past_designs: Tensor | None = None, past_outcomes: Tensor | None = None
    ) -> Tensor:
        batch_shape = past_designs.shape[:-2] if past_designs is not None else (1,)
        if self.design_dist is None:
            design = dist.Normal(self.design_mean, 1.0).sample(batch_shape)  # type: ignore
        else:
            design = self.design_dist.sample(batch_shape)  # type: ignore

        return design


class EmpiricalPrior:
    def __init__(self, samples: Tensor):
        self.samples = samples
        self.N = len(self.samples)

    def sample(self, shape: torch.Size) -> Tensor:
        # sample randomly from the given samples
        idx = torch.randint(0, self.N, shape)
        return self.samples[idx]


class NestedMonteCarlo(nn.Module):
    def __init__(
        self,
        prior: dist.Distribution,
        outcome_likelihood: Callable,  # returns a distribution when called with params and designs
        proposal: Callable | None = None,  # returns a dist when called
        num_inner_samples: int = 512,
    ):
        """
        Args:
            prior: dist.Distribution
                Prior distribution for the parameters.
            outcome_likelihood: Callable
                Outcome likelihood function - should return a distribution when called with params and designs
            proposal: Callable | None
                Importance sampling proposal distribution for the parameters;
                Optional, if None, we use the prior
            num_inner_samples: int
                The number of samples to use to approximate the marginal.
            lower_bound: bool
                Whether to use a lower bound on the marginal.
        """
        super().__init__()
        self.prior = prior
        if proposal is not None:
            assert proposal is not None
        self.proposal = proposal
        self.outcome_likelihood = outcome_likelihood
        self.num_inner_samples = num_inner_samples

    def get_log_likelihood_and_marginal(
        self, params: Tensor, designs: Tensor, outcomes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the log likelihood and marginal for the given parameters, designs, and outcomes.

        Args:
            params: Tensor of shape [B, K, p]
            designs: Tensor of shape [B, p, T]
            outcomes: Tensor of shape [B, 1, T]

        Returns:
            log_likelihood: Tensor of shape [B, 1]
            log_marginal_lb: Tensor of shape [B, 1]
            log_marginal_ub: Tensor of shape [B, 1]
        """
        batch_size = params.shape[0]  # params should be shape [B, K, p]
        if self.proposal is not None:
            proposal = self.proposal(designs, outcomes)
            reweight = True
            inner_params_samples = proposal.sample((self.num_inner_samples,))
        else:  # use the prior
            proposal = self.prior
            reweight = False
            inner_params_samples = proposal.sample(
                torch.Size([self.num_inner_samples, batch_size])
            )  # [N, B, K, p]

        # combine inner and outer samples of log_k and alpha
        params = torch.cat(
            [params.unsqueeze(0), inner_params_samples], dim=0
        )  # [N+1, B, K, p]

        # Compute the log likelihood under all params:
        # Reshape designs and outcomes to match the new parameter shapes
        designs_expanded = designs.unsqueeze(0).expand(
            self.num_inner_samples + 1, *designs.shape
        )  # [B, p, T] -> [N+1, B, p, T]
        outcomes_expanded = outcomes.unsqueeze(0).expand(
            self.num_inner_samples + 1, *outcomes.shape
        )  # [B, 1, T] -> [N+1, B, 1, T]

        # Compute log likelihood for all samples and all design-outcome pairs
        log_likelihood = (
            self.outcome_likelihood(
                # treat T as another batch dim and sum at the end
                params.unsqueeze(0),  # [1, N+1, B, K, p]
                designs_expanded.permute(3, 0, 1, 2),  # [T, N+1, B, p]
            )
            .log_prob(outcomes_expanded.permute(3, 0, 1, 2))  # [T, N+1, B, 1]
            .sum(0)  # [N+1, B, 1]
        )

        # compute log p() - log q()  if reweighting
        if reweight:
            # both logprobs should be shape [N+1, B]
            logprobs_under_p = self.prior.log_prob(params)
            logprobs_under_q = proposal.log_prob(params)
            # check!
            log_weights = logprobs_under_p - logprobs_under_q  # [N+1, B]
        else:
            log_weights = torch.zeros_like(log_likelihood)  # [N+1, B, 1]

        # log-sum rather than log-mean-exp. constant added back in .estimate only
        # for the PCE (lower bound):
        log_marginal_lb = torch.logsumexp(log_likelihood + log_weights, dim=0)  # [B, 1]

        # for the NMC (upper bound) remove the first sample from the log-sum-exp
        log_marginal_ub = torch.logsumexp(
            log_likelihood[1:] + log_weights[1:], dim=0
        )  # [B, 1]

        # the log likelihood for the original parameters (first row)
        # [B, 1], [B, 1]
        return log_likelihood[0], log_marginal_lb, log_marginal_ub

    @torch.no_grad()
    def estimate(
        self, params: Tensor, designs: Tensor, outcomes: Tensor
    ) -> dict[str, tuple[float, float]]:
        """
        Estimate the mutual information between params and outcomes (EIG)
        I(params; outcomes) = \E_{p(params)p(outcomes|params, designs)} [
            log p(outcomes|params, designs) - log p(outcomes|designs)
        ]

        Note that this is an expectation over the parameters and designs.
        The designs are being optimised over so we need to be careful with the expectation -->
            - Reparameterisation trick where .has_rsample() is True
            - REINFORCE gradient estimator otherwise
        For this model we use the reparameterisation trick.

        params: Tensor of shape [..., K, p]
        designs: Tensor of shape [..., p, T]
        outcomes: Tensor of shape [..., 1, T]

        If ... is 2-dim, will return mean and standard error over the leftmost dimension.

        returns: float
        """
        if params.ndim == 3:
            assert designs.ndim == 3 and outcomes.ndim == 3
            params = params.unsqueeze(0)
            designs = designs.unsqueeze(0)
            outcomes = outcomes.unsqueeze(0)
        reps = params.shape[0]
        mi_lb = torch.empty(reps)
        mi_ub = torch.empty(reps)

        for i, (theta, xi, y) in tqdm(
            enumerate(zip(params, designs, outcomes)), total=reps
        ):
            log_likelihood, log_marginal_lb, log_marginal_ub = (
                self.get_log_likelihood_and_marginal(theta, xi, y)
            )
            mi_estimate_lb = (log_likelihood - log_marginal_lb).mean(0) + math.log(
                self.num_inner_samples
            )
            mi_estimate_ub = (log_likelihood - log_marginal_ub).mean(0) + math.log(
                self.num_inner_samples
            )
            mi_lb[i] = mi_estimate_lb
            mi_ub[i] = mi_estimate_ub

        return {
            "lb": (mi_lb.mean(0).item(), mi_lb.std(0).item() / math.sqrt(reps)),
            "ub": (mi_ub.mean(0).item(), mi_ub.std(0).item() / math.sqrt(reps)),
        }

    def differentiable_loss(
        self, params: Tensor, designs: Tensor, outcomes: Tensor
    ) -> tuple[Tensor, Tensor]:
        # implement the differentiable loss using REINFORCE (aka score) gradient estimator
        log_likelihood, log_marginal_lb, log_marginal_ub = (
            self.get_log_likelihood_and_marginal(params, designs, outcomes)
        )
        diff_loss_lb = -(log_likelihood - log_marginal_lb).mean()
        diff_loss_ub = -(log_likelihood - log_marginal_ub).mean()
        return diff_loss_lb, diff_loss_ub


# Myopic design strategy:
def run_myopic_design(
    model: LocationFinding,
    true_theta: Tensor,
    num_designs: int,
    num_grad_steps: int = 1000,
    train_batch_shape: torch.Size = torch.Size([64]),
    symmetrise: bool = False,
    plot: bool = False,
) -> dict[str, Tensor]:
    designs_so_far = []
    outcomes_so_far = []
    hmc_posterior_samples = None

    for t in tqdm(range(num_designs), desc="Designing"):
        with torch.enable_grad():
            objective = NestedMonteCarlo(
                prior=model.prior(),
                outcome_likelihood=model.outcome_likelihood,
                num_inner_samples=512,
            )
            step_model = LocationFinding(
                K=model.K,
                p=model.p,
                T=1,
                design_func=StaticDesign(
                    designs=torch.randn(1, model.p) * 1.0, learn_designs=True
                ),
                prior_samples=(hmc_posterior_samples.squeeze(0) if t > 0 else None),
            )
            # learn the design:
            optim = torch.optim.AdamW(step_model.design_func.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=num_grad_steps, eta_min=1e-4
            )
            for i in range(num_grad_steps):
                optim.zero_grad()
                train_batch = step_model(batch_shape=train_batch_shape)
                loss, _ = objective.differentiable_loss(**train_batch)
                loss.backward()
                optim.step()
                scheduler.step()

        # execute the design, store the design and the outcome
        experiment_result = step_model.run_policy(true_theta)
        designs_so_far.append(experiment_result["designs"])
        outcomes_so_far.append(experiment_result["outcomes"])

        hmc_posterior_samples = step_model.run_hmc_posterior(
            designs=torch.cat(designs_so_far, dim=-1),
            outcomes=torch.cat(outcomes_so_far, dim=-1),
            num_chains=4,
            num_samples=1000,
            symmetrise=symmetrise,
        )  # [1, 4000, K, p]

    # at end of loop, plot the realisations and the hmc posterior samples
    if plot:
        plot_posterior_comparison(
            true_theta=true_theta,
            amortised_samples=None,
            hmc_samples=hmc_posterior_samples,
            prior_samples=step_model.prior().sample((10000,)),
        )
        model.plot_realisations(
            params=true_theta,
            designs=torch.cat(designs_so_far, dim=-1),
            outcomes=torch.cat(outcomes_so_far, dim=-1),
        )

    return {
        "designs": torch.cat(designs_so_far, dim=-1),
        "outcomes": torch.cat(outcomes_so_far, dim=-1),
        "params": true_theta,
    }


def run_myopic_ba_design(
    model: LocationFinding,
    true_theta: Tensor,
    num_designs: int,
    num_grad_steps: int = 1000,
    train_batch_shape: torch.Size = torch.Size([64]),
    symmetrise: bool = False,
    plot: bool = False,
) -> dict[str, Tensor]:
    designs_so_far = []
    outcomes_so_far = []
    posterior_samples = None
    num_params = model.K * model.p
    npost_samples = 10000

    for t in range(num_designs):
        step_model = LocationFinding(
            K=model.K,
            p=model.p,
            T=1,
            design_func=StaticDesign(
                designs=torch.randn(1, model.p), learn_designs=True
            ),
            prior_samples=(
                posterior_samples.squeeze(0)
                if t > 0 and posterior_samples is not None
                else None
            ),
        )
        print("initial designs:", step_model.design_func.designs)
        if t == 0:
            step_model.design_func.learn_designs = False
            step_model.design_func.designs = torch.nn.Parameter(
                torch.randn(1, model.p) * 0.01
            )
        else:
            with torch.enable_grad():
                posterior_net_for_design = bf.networks.CouplingFlow()
                posterior_net_for_design.build(
                    xz_shape=(64, num_params), conditions_shape=(64, model.p + 1)
                )
                # Joint optimization
                optim_joint = torch.optim.AdamW(
                    [
                        {"params": step_model.design_func.parameters(), "lr": 1e-2},
                        {"params": posterior_net_for_design.parameters(), "lr": 1e-3},
                    ]
                )
                scheduler_joint = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim_joint, T_max=num_grad_steps, eta_min=1e-8
                )

                pbar = trange(
                    num_grad_steps, desc="Joint Optimization", leave=False, position=0
                )
                for i in pbar:
                    optim_joint.zero_grad()
                    theta, designs, outcomes = step_model(
                        batch_shape=train_batch_shape
                    ).values()
                    design_obs_pairs_flattened = torch.cat(
                        [designs.detach(), outcomes], dim=-2
                    ).flatten(-2)
                    logprobs = posterior_net_for_design.log_prob(
                        theta.flatten(-2), conditions=design_obs_pairs_flattened
                    )
                    loss = -logprobs.mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(posterior_net_for_design.parameters())
                        + list(step_model.design_func.parameters()),
                        max_norm=1.0,
                    )
                    optim_joint.step()
                    scheduler_joint.step()
                    pbar.set_description(f"Joint Step {i}, loss: {loss.item()}")
                    pbar.update(1)

        print("final designs:", step_model.design_func.designs)
        experiment_result = step_model.run_policy(true_theta)
        designs_so_far.append(experiment_result["designs"])
        outcomes_so_far.append(experiment_result["outcomes"])
        #####################
        # Train amortized posterior with all designs and outcomes so far
        posterior_net_final = bf.networks.CouplingFlow()
        posterior_net_final.build(
            xz_shape=(64, num_params),
            conditions_shape=(64, (model.p + 1) * (t + 1)),
        )

        optim_final = torch.optim.AdamW(posterior_net_final.parameters(), lr=1e-3)
        scheduler_final = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_final, T_max=num_grad_steps, eta_min=1e-6
        )

        pbar = trange(
            num_grad_steps, desc="Training Final Posterior", leave=False, position=0
        )
        all_designs: Tensor = torch.cat(designs_so_far, dim=-1)  # [B=1, p, t]
        all_outcomes: Tensor = torch.cat(outcomes_so_far, dim=-1)  # [B=1, 1, t]
        print("all_designs shape:", all_designs.shape)
        print("all_outcomes shape:", all_outcomes.shape)

        # Create a LocationFinding model with static designs
        assert all_designs.shape[-1] == t + 1
        static_model = LocationFinding(
            K=model.K,
            p=model.p,
            T=t + 1,
            design_func=StaticDesign(
                designs=all_designs.squeeze(0).T,  # [p, t] -> [t, p]
                learn_designs=False,
            ),
        )
        assert static_model.design_func.designs.shape == (t + 1, model.p)
        for i in pbar:
            with torch.enable_grad():
                optim_final.zero_grad()
                # theta are [B, K, p]; designs are [B, p, T]; outcomes are [B, 1, T]
                theta, designs, outcomes = static_model(
                    batch_shape=train_batch_shape
                ).values()

                # Flatten the design-outcome pairs
                design_obs_pairs_flattened = torch.cat(
                    [designs, outcomes], dim=-2
                ).flatten(-2)
                # Compute log probabilities
                logprobs = posterior_net_final.log_prob(
                    theta.flatten(-2), conditions=design_obs_pairs_flattened
                )
                loss = -logprobs.mean()
                loss.backward()
                optim_final.step()
                scheduler_final.step()
                pbar.set_description(f"Final Posterior Step {i}, loss: {loss.item()}")
                pbar.update(1)

        posterior_samples = torch.stack(
            [
                posterior_net_final.sample(
                    (npost_samples,),
                    conditions=torch.cat([all_designs, all_outcomes], dim=-2)
                    .flatten(-2)
                    .expand(npost_samples, -1),
                )
            ],
            dim=0,
        )
        posterior_samples = posterior_samples.reshape(
            *posterior_samples.shape[:-1], model.K, model.p
        )
        if symmetrise:
            posterior_samples = posterior_samples.reshape(
                1, npost_samples * model.K, model.p
            )
            posterior_samples = posterior_samples.unsqueeze(-2).expand(
                1, npost_samples * model.K, model.K, model.p
            )
        print("Nan posterior samples:", posterior_samples.isnan().sum())
        posterior_samples = posterior_samples.nan_to_num()
        posterior_samples = torch.clip(posterior_samples, -5, 5)

        hmc_posterior_samples = step_model.run_hmc_posterior(
            designs=torch.cat(designs_so_far, dim=-1),
            outcomes=torch.cat(outcomes_so_far, dim=-1),
            num_chains=4,
            num_samples=1000,
            symmetrise=symmetrise,
        )

        plot_posterior_comparison(
            true_theta=true_theta,
            amortised_samples=posterior_samples,
            hmc_samples=hmc_posterior_samples,
            prior_samples=step_model.prior().sample(torch.Size([10000])),
        )

    if plot:
        plot_posterior_comparison(
            true_theta=true_theta,
            amortised_samples=posterior_samples,
            hmc_samples=None,
            prior_samples=step_model.prior().sample(torch.Size([10000])),
        )
        model.plot_realisations(
            params=true_theta,
            designs=torch.cat(designs_so_far, dim=-1),
            outcomes=torch.cat(outcomes_so_far, dim=-1),
        )

    return {
        "designs": torch.cat(designs_so_far, dim=-1),
        "outcomes": torch.cat(outcomes_so_far, dim=-1),
        "params": true_theta,
    }


# if __name__ == "__main__":
#     # run small sanity checks:
#     test_batch_shape = (4,)
#     eval_batch_shape = (8, 4)  # to eval the MIs

#     K = 2
#     num_designs = 30
#     p = 2
#     initial_designs = torch.randn(num_designs, p)
#     design_strategies = {
#         "static": StaticDesign(designs=initial_designs, learn_designs=True),
#         "random": RandomDesign(design_shape=(p,)),
#     }

#     for design_strategy_name, design_strategy in design_strategies.items():
#         torch.manual_seed(20241011)
#         print(f"Design Strategy: {design_strategy_name}")
#         _model = LocationFinding(K=K, p=p, T=num_designs, design_func=design_strategy)
#         test_sims = _model(batch_shape=test_batch_shape)
#         assert test_sims["designs"].shape == (*test_batch_shape, p, num_designs)
#         assert test_sims["outcomes"].shape == (*test_batch_shape, 1, num_designs)
#         assert test_sims["params"].shape == (*test_batch_shape, K, p)
#         _model.plot_realisations(**test_sims)

#         nmc_eval = NestedMonteCarlo(
#             prior=_model.prior(),
#             outcome_likelihood=_model.outcome_likelihood,
#             num_inner_samples=50000,
#         )
#         eval_dataset = _model(batch_shape=eval_batch_shape)
#         eig_estimates = nmc_eval.estimate(**eval_dataset)
#         print(
#             f"Design Strategy={design_strategy_name}, evaluation: \nLower bound: {eig_estimates['lb']}; Upper bound: {eig_estimates['ub']}"
#         )

# train an initial posterior using random designs
# num_grad_steps = 1000
# train_batch_shape = (64,)
# K, p = 2, 2
# _model = LocationFinding(K=K, p=p, T=1, design_func=RandomDesign(design_shape=(p,)))
# num_params = _model.K * _model.p
# _posterior_net = bf.networks.CouplingFlow()
# _posterior_net.build(xz_shape=(64, num_params), conditions_shape=(64, _model.p + 1))
# optim = torch.optim.AdamW(_posterior_net.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optim, T_max=num_grad_steps, eta_min=1e-8
# )
# pbar = trange(num_grad_steps, desc="Training", leave=False, position=0)
# with torch.enable_grad():
#     for i in pbar:
#         optim.zero_grad()
#         theta, designs, outcomes = _model(batch_shape=train_batch_shape).values()
#         design_obs_pairs_flattened = torch.cat([designs, outcomes], dim=-2).flatten(-2)
#         logprobs = _posterior_net.log_prob(
#             theta.flatten(-2), conditions=design_obs_pairs_flattened
#         )
#         loss = -logprobs.mean()
#         loss.backward()
#         optim.step()
#         scheduler.step()
#         pbar.set_description(f"Step {i}, loss: {loss.item()}")
#         pbar.update(1)


torch.manual_seed(20241011)
# _model = LocationFinding(K=2, p=2, T=1, design_func=lambda x: x)
run_myopic_ba_design(
    # pretrained_posterior_net=_posterior_net,
    model=LocationFinding(K=2, p=2, T=1, design_func=lambda x: x),
    true_theta=torch.randn(1, 2, 2),
    num_designs=1,
    num_grad_steps=5000,
    symmetrise=True,
    plot=True,
)

# let's train an amortised posterior with a summary network
summary_net = bf.networks.SetTransformer(summary_dim=16)
