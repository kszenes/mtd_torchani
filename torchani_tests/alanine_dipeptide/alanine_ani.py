# %%
import sys
# sys.path.append("/Users/kalmanszenes/code/mtd_torchani/torchmd")
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

import cProfile
import pstats

# np.random.seed(0)
# torch.manual_seed(0)
# structure = sys.argv[1]
structure = 'dialaB.pdb'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
precision = torch.float32

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
# print(system_ani.forces)
# print(system_ani.pos)

psi_list = [6, 8, 14, 16]
phi_list = [4, 6, 8, 14]

# alanine_ase.get_dihedrals([psi_list, phi_list])


# print(system_ani.get_dijj
# print(system_ani.get_dihedrals_ani() * 180 / np.pi)
# print(system_ani.get_phi())
# print(system_ani.get_positions())


# %%
# height = 0.004336
# width = 0.05

# phi = system_ani.get_phi()
# peaks = torch.tensor([phi.detach() + 0.1, phi.detach() + 0.2])

# bias = system_ani.get_bias(phi, peaks, height=1, width=1)
# bias
# # bias = system_ani.get_bias(dihedrals, dihedrals.detach() + offset, height=0.004336, width=0.05) + system_ani.get_bias(dihedrals, dihedrals.detach() - 2 * offset, height=0.004336, width=0.05)

# %%
# f_bias = -torch.autograd.grad(bias.sum(), system_ani.pos)[0]
# f_bias

# %%
from minimizers import minimize_pytorch_bfgs_ANI
print(system_ani.pos)
minimize_pytorch_bfgs_ANI(system_ani, steps=1000)
print(system_ani.pos)

#%%
# ---------- Torchani ---------
from integrator import Integrator_ANI, Langevin_integrator
from ase.units import eV, Hartree, kB

langevin_temperature = 300  # K
langevin_gamma = 0.2
timestep = 1  # fs
height=0.004336
width=0.05


integrator_ani = Langevin_integrator(system_ani, timestep, device, fr=langevin_gamma, temp=langevin_temperature, height=height, width=width)

# %%
n_iter = int(1e2)

# %%
with cProfile.Profile() as pr:
  integrator_ani.run(n_iter, device=device)

# %%
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='profiling.prog')
# %%
n_iter = int(1e2)
print_iter = 1

# cProfile.run('integrator_ani.run(n_iter, device=device)')

integrator_ani.run(n_iter, traj_file='log/' + structure.split('.')[0] + '.xyz', log_file='log/' + structure.split('.')[0] + '.csv', log_interval=print_iter, device=device, metadyn=True)


# %%
free_e = integrator_ani.get_free_energy()
# # torch.sub(torch.arange(-np.pi, np.pi, 2*np.pi/1000), integrator_ani.peaks)
# # width=0.05
# # height=0.004336
# # x_range = torch.arange(-np.pi, np.pi, 2*np.pi/1000)
# # gauss = - height * torch.sum(torch.exp(-(x_range - integrator_ani.peaks[:,None])**2 / (2*width**2)), dim=0)
# # import matplotlib.pyplot as plt
# # plt.plot(x_range, gauss)

# %%
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('log/' + structure.split('.')[0] + '.csv', skiprows=1)
# %%
plt.scatter(df['Psi'][::100], df['Phi'][::100], marker='.')
plt.xlabel('$Phi$ / rad')
plt.ylabel('$Psi$ / rad')


# %%
# import nglview as nv
# from ase.io import read
# nv.show_asetraj(read('log/dialaA.xyz', index=':'), gui=True)
# # %%
# import pandas as pd
# df = pd.read_csv('log/dialaA.csv', skiprows=1)
# df
# # %%
# import matplotlib.pyplot as plt
# plt.scatter(df['Phi'], df['Psi'], marker='.')
# plt.xlabel('Phi degrees')
# plt.ylabel('Psi degrees')
# # %%
# df['Phi'].plot()
# # %%
# df['Psi'].plot()
# # %%

# %%
