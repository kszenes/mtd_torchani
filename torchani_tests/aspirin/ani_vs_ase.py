# %%
sys.path.append("/Users/kalmanszenes/code/mtd_torchani/torchmd")

import torch
from torchmd.integrator import maxwell_boltzmann, kinetic_energy, kinetic_to_temp
from systems import System_ANI
import torchani
import numpy as np

np.random.seed(0)
torch.manual_seed(0)


import torchani
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision = torch.float

model = torchani.models.ANI1ccx(periodic_table_index=True).to(device) # ASE
# model = torchani.models.ANI2x(periodic_table_index=True).to(device) # non-ASE


coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                             requires_grad=True, device=device)

# In periodic table, C = 6 and H = 1
species = torch.tensor([[6, 1, 1, 1, 1]], device=device)

masses = torchani.utils.get_atomic_masses(species).reshape(-1, 1).to(device)

system_ani = System_ANI(model, coordinates, species, masses, coordinates.shape[1], nreplicas=1, precision=precision, device=device)
# system_ani.set_positions(coordinates) # expects (1, n_atoms, coords)
# system_ani.set_species(species)
# system.set_box(mol.box)
system_ani.set_velocities(maxwell_boltzmann(system_ani.masses, T=300, replicas=1))
print(f'Ekin = {kinetic_to_temp(kinetic_energy(system_ani.masses, system_ani.vel), len(system_ani.masses))}')
# system_ani.set_masses(masses)
print(system_ani.energy, "\n", torch.max(system_ani.forces), torch.min(system_ani.forces))
print(system_ani.forces)
print(system_ani.pos)

# %%
from minimizers import minimize_pytorch_bfgs_ANI
minimize_pytorch_bfgs_ANI(system_ani, steps=1000)
print(system_ani.pos)

#%%

from integrator import Integrator_ANI
from ase.units import eV, Hartree, kB

langevin_temperature = 300  # K
langevin_gamma = 0.002
timestep = 1  # fs

verbose = False

integrator_ani = Integrator_ANI(system_ani, timestep, device, gamma=langevin_gamma, T=langevin_temperature)
print("E\tT")
with open('ch4_traj.xyz', 'w') as f:
  for i in range(2000):
    Ekin, Epot, T = integrator_ani.step()
    system_ani.compute_forces()

    if verbose:
      print(system_ani.energy, Ekin, T)
      print(system_ani.pos)
      print(torch.max(system_ani.forces), torch.min(system_ani.forces))
    else:
      if i % 100 == 99:
        print(system_ani.energy, Ekin, T)
        print(system_ani.pos)
        print(torch.max(system_ani.forces), torch.min(system_ani.forces))

  # else if i % 50 == 49:
  #   print(system_ani.energy, Ekin, T)
  #   print(system_ani.pos)
  #   print(system_ani.forces)


# %%
# ASE
import ase
from ase.units import eV, Hartree
import torchani
import torch

model = torchani.models.ANI1ccx(periodic_table_index=True)

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True)

ch4 = ase.Atoms('CH4', coordinates.detach().numpy().reshape(-1, 3))
ch4.set_calculator(model.ase())
print(ch4.get_potential_energy(), "\n",  np.max(ch4.get_forces()), np.min(ch4.get_forces()))

# %%
# ASE
from ase.optimize import LBFGS
dyn = LBFGS(ch4, trajectory='ase.traj')
dyn.run()
print(ch4.get_potential_energy() * eV / Hartree)
print(ch4.get_positions())

# %%
# ASE
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory

def printenergy(a=ch4):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
    # print(a.positions)
    print(np.max(a.get_forces()), np.min(a.get_forces()))

dyn = Langevin(ch4, 1 * units.fs, 300 * units.kB, 0.2)
dyn.attach(printenergy, interval=1)

traj = Trajectory('ase_ch4.traj', 'w', ch4)
dyn.attach(traj.write, interval=100)
print("Beginning dynamics...")
printenergy()
dyn.run(5000)
# %%
# EMT
"""Demonstrates molecular dynamics with constant temperature."""
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units

from asap3 import EMT  # Way too slow with ase.EMT !

import torch
import ase
T = 300

coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True)

atoms = ase.Atoms('CH4', coordinates.detach().numpy().reshape(-1, 3))

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = EMT()

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 5 * units.fs, T * units.kB, 0.002)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory('moldyn3.traj', 'w', atoms)
dyn.attach(traj.write, interval=50)

# Now run the dynamics
printenergy()
dyn.run(5000)
# %%
# ------- TorchAni -----------

atom_types = ['C', 'H', 'H', 'H', 'H']

with open('ch4_traj.xyz', 'w') as f:
  for i in range(len(traj)):
    coord = traj[i].reshape(-1, 3)
    f.write(str(coord.shape[0]) + '\n\n')
    for j in range(coord.shape[0]):
      f.write(atom_types[j])
      for k in range(3):
        f.write(' ' + str(coord[j, k])) 
      f.write('\n')


# %%
import nglview as nv
def view_trajectory(system):
    t2 = nv.ASETrajectory(system)
    w2 = nv.NGLWidget(t2, gui=True)
    #w2.add_spacefill()
    return w2
# %%
from ase.io import read
view_trajectory(read('ch4_traj.xyz', index=':'))
# %%
