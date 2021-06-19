# %%
sys.path.append("/Users/kalmanszenes/code/mtd_torchani/torchmd")

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision = torch.float

model = torchani.models.ANI1ccx(periodic_table_index=True).to(device) # ASE
# model = torchani.models.ANI2x(periodic_table_index=True).to(device) # non-ASE

alanine_ase = ase.Atoms(read('alanine_dipeptide.xyz'))
alanine_ase.set_calculator(model.ase())
# aspirin_ase.set_velocities(maxwell_boltzmann(system_ani.masses, T=300, replicas=1))
alanine_ase.get_kinetic_energy()


system_ani = System_ANI.from_ase(alanine_ase)

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

# %%
from minimizers import minimize_pytorch_bfgs_ANI
minimize_pytorch_bfgs_ANI(system_ani, steps=1000)
print(system_ani.pos)

#%%
# ---------- Torchani ---------
from integrator import Integrator_ANI, Langevin_integrator
from ase.units import eV, Hartree, kB

langevin_temperature = 300  # K
langevin_gamma = 0.2
timestep = 1  # fs

integrator_ani = Langevin_integrator(system_ani, timestep, device, fr=langevin_gamma, temp=langevin_temperature)

# %%
n_iter = int(1000)
print_iter = 10

# cProfile.run('integrator_ani.run(n_iter, device=device)')

integrator_ani.run(n_iter, traj_file='alanine_ani.xyz', traj_interval=print_iter,
                    log_file='alanine_ani.csv', log_interval=print_iter, device=device)

# %%
nv.show_asetraj(read('alanine_ani.xyz', index=':'), gui=True)
# %%
import pandas as pd
df = pd.read_csv('alanine_ani.csv', skiprows=1)
df
# %%