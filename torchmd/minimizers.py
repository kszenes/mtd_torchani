import torch
import numpy as np

from torchani.units import hartree2ev

def minimize_pytorch_lbfgs_ANI(system,steps=1000):
    '''Relax structure to minimum'''
    if steps == 0:
        return

    # opt = torch.optim.LBFGS([system.pos], max_iter=steps, tolerance_change=1e-09)
    pos = system.get_positions()
    opt = torch.optim.LBFGS([pos], max_iter=steps, tolerance_change=1e-09)

    def closure(step):
        opt.zero_grad()
        E = hartree2ev(system.model((system.species, pos)).energies)
        system.energy = E
        E.backward()
        maxforce = float(torch.max(torch.norm(system.pos.grad, dim=1)))
        print("{0:4d}   {1:3.12f}  {2:3.12f}   {3: 3.12f}".format(step[0], float(E), float(E)/27.211386246, maxforce))
        step[0] += 1
        return E

    print("{0:4s} {1:9s} {2:9s}  {3:9s}".format("Iter", "\t E (eV)", "\t\t E (a.u.)",  "\t\t fmax"))
    step = [0]
    opt.step(lambda: closure(step))

    # system.pos[:] = pos.detach().requires_grad_(False)
    system.set_positions(pos)
