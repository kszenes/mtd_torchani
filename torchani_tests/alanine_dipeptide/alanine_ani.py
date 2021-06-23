# %%
import sys
sys.path.append("/data/kszenes/mtd_torchani/torchmd")

import torch
# from torchmd.integrator import maxwell_boltzmann, kinetic_energy, kinetic_to_temp

from systems import System_ANI
from integrator import maxwell_boltzmann
import torchani
import numpy as np
import nglview as nv

import ase
from ase.io import read

# import cProfile

# np.random.seed(0)
# torch.manual_seed(0)
structure = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
precision = torch.float

model = torchani.models.ANI1ccx(periodic_table_index=True).to(device) # ASE
# model = torchani.models.ANI2x(periodic_table_index=True).to(device) # non-ASE

alanine_ase = ase.Atoms(read(structure))
alanine_ase.set_calculator(model.ase())
# aspirin_ase.set_velocities(maxwell_boltzmann(system_ani.masses, T=300, replicas=1))
alanine_ase.get_kinetic_energy()


system_ani = System_ANI.from_ase(alanine_ase, device=device)

system_ani.set_velocities(maxwell_boltzmann(system_ani.masses, T=300, replicas=1, device=device))
print(system_ani.get_kinetic_energy())
print(system_ani.get_temperature())
# print(f'Ekin = {kinetic_to_temp(kinetic_energy(system_ani.masses, system_ani.vel), len(system_ani.masses))}')
# # system_ani.set_masses(masses)
# print(system_ani.energy, "\n", torch.max(system_ani.forces), torch.min(system_ani.forces))
print(system_ani.forces)
# print(system_ani.pos)

psi = [7, 6, 8, 10] # Nitrogen
phi = [15, 14, 8, 10] # Oxygen

print(system_ani.get_dihedrals())
print(system_ani.get_dihedrals_ani())

# # %%
# # --------- Dihedrals testing ----------
# dihedrals = torch.tensor([1, 2])
# offset = torch.tensor([[1, -1], [-1, 3]])
# torch.exp(dihedrals + offset)

# # %%
# height = 0.004336
# width = 0.05

# dihedrals = system_ani.get_dihedrals_ani()
# offset = torch.tensor([[1, 2]])
# offset_2 = torch.tensor([[2, 1]])
# peaks = torch.stack((dihedrals.detach() + offset, dihedrals.detach() + offset_2), axis=1)
# # peaks = dihedrals.detach() + offset

# bias = system_ani.get_bias(dihedrals, peaks, height=height, width=width)
# bias
# # bias = system_ani.get_bias(dihedrals, dihedrals.detach() + offset, height=0.004336, width=0.05) + system_ani.get_bias(dihedrals, dihedrals.detach() - 2 * offset, height=0.004336, width=0.05)

# # %%
# f_bias = -torch.autograd.grad(bias.sum(), system_ani.pos)[0]
# f_bias

# %%
from minimizers import minimize_pytorch_bfgs_ANI
print(system_ani.pos)
#minimize_pytorch_bfgs_ANI(system_ani, steps=1000)
#print(system_ani.pos)

#%%
# ---------- Torchani ---------
from integrator import Integrator_ANI, Langevin_integrator
from ase.units import eV, Hartree, kB

langevin_temperature = 300  # K
langevin_gamma = 0.2
timestep = 1  # fs

integrator_ani = Langevin_integrator(system_ani, timestep, device, fr=langevin_gamma, temp=langevin_temperature)

# %%
n_iter = int(1e4)
print_iter = 1

# cProfile.run('integrator_ani.run(n_iter, device=device)')

integrator_ani.run(n_iter, log_file='log/' + sys.argv[1] + '.csv', log_interval=print_iter, device=device)

# %%
# import nglview as nv
# from ase.io import read
# nv.show_asetraj(read('alanine_ani.xyz', index=':'), gui=True)
# %%
# import pandas as pd
# df = pd.read_csv('alanine_ani.csv', skiprows=1)
# df
# # %%
# import matplotlib.pyplot as plt
# plt.scatter(df['Phi'], df['Psi'], marker='.')
# plt.xlabel('Phi degrees')
# plt.ylabel('Psi degrees')
# %%
# %%
