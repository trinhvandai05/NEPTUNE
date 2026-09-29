"""The angular (shape) flow.

    zdot_k = -P_{z_k} grad [ V + Psi + gamma * sg[Z] * sum_l p_l kappa_h(., z_l) ]

In the H1 slice Z == 1, so gamma * sg[Z] = gamma.  The single surviving factor
of Z on the interaction term is the warp factor of the Hellinger-Kantorovich
cone; it is written here as a plain `gamma` precisely because Z does not exist
in this slice.

The three gradient terms are computed separately (`components`) so
diagnostics can inspect each -- in particular V vs Psi separability (D14).
"""

from __future__ import annotations

import torch

from ..state.kernel import kernel, kernel_ambient_grad, project
from ..state.satiation import AtomBank, contributions, psi_from_contrib


class AngularField:
    def __init__(self, potential, u, w, e_atom: torch.Tensor, bank: AtomBank,
                 gamma: float, h: float, sigma: float, use_satiation: bool):
        self.potential, self.u, self.w = potential, u, w
        self.e_atom, self.bank = e_atom, bank
        self.gamma, self.h, self.sigma = gamma, h, sigma
        self.use_satiation = use_satiation

    def components(self, z, p, now):
        gv = project(z, self.potential.ambient_grad(z, self.u, self.w))
        if self.use_satiation:
            _, gpsi = psi_from_contrib(z, self.e_atom, contributions(self.bank, now),
                                       self.sigma, want_grad=True)
            gpsi = project(z, gpsi)
        else:
            gpsi = torch.zeros_like(z)
        gint = project(z, self.gamma * kernel_ambient_grad(z, z, p, self.h))
        return {"grad_V": gv, "grad_Psi": gpsi, "grad_int": gint}

    def __call__(self, z, p, now):
        c = self.components(z, p, now)
        return -(c["grad_V"] + c["grad_Psi"] + c["grad_int"])

    def energy_terms(self, z, p, now):
        """First variation f_k at each particle (drives the replicator). [B, M]"""
        f = self.potential.value(z, self.u, self.w)
        if self.use_satiation:
            f = f + psi_from_contrib(z, self.e_atom, contributions(self.bank, now), self.sigma)[0]
        return f + self.gamma * torch.einsum("bkl,bl->bk", kernel(z, z, self.h), p)

    def free_energy(self, z, p, now):
        """F_hat on the particle system. [B]

        Closed-form Renyi-2 interaction, never a Shannon-entropy estimate: KDE
        entropy from M~16 particles in d=64 is statistically indefensible, and
        the interaction term already IS the Renyi-2 collision energy.
        """
        lin = self.potential.value(z, self.u, self.w)
        if self.use_satiation:
            lin = lin + psi_from_contrib(z, self.e_atom, contributions(self.bank, now), self.sigma)[0]
        quad = torch.einsum("bk,bkl,bl->b", p, kernel(z, z, self.h), p)
        return (lin * p).sum(-1) + 0.5 * self.gamma * quad
