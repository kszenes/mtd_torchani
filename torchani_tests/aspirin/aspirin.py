# %%
sys.path.append("/Users/kalmanszenes/code/mtd_torchani/torchmd")

import torch
# from torchmd.integrator import maxwell_boltzmann, kinetic_energy, kinetic_to_temp

from systems import System_ANI
from integrator import maxwell_boltzmann
import torchani
import numpy as np


# np.random.seed(0)
# torch.manual_seed(0)


import torchani
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision = torch.float

model = torchani.models.ANI1ccx(periodic_table_index=True).to(device) # ASE
# model = torchani.models.ANI2x(periodic_table_index=True).to(device) # non-ASE

import ase
from ase.io import read
aspirin_ase = ase.Atoms(read('aspirin.xyz'))
aspirin_ase.set_calculator(model.ase())
# aspirin_ase.set_velocities(maxwell_boltzmann(system_ani.masses, T=300, replicas=1))
aspirin_ase.get_kinetic_energy()


system_ani = System_ANI.from_ase(aspirin_ase)

system_ani.set_velocities(maxwell_boltzmann(system_ani.masses, T=300, replicas=1))
system_ani.get_kinetic_energy()
system_ani.get_temperature()
# print(f'Ekin = {kinetic_to_temp(kinetic_energy(system_ani.masses, system_ani.vel), len(system_ani.masses))}')
# # system_ani.set_masses(masses)
# print(system_ani.energy, "\n", torch.max(system_ani.forces), torch.min(system_ani.forces))
# print(system_ani.forces)
# print(system_ani.pos)

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
n_iter = 10000
integrator_ani.run(n_iter, traj_file='aspirin_ani.xyz', traj_interval=n_iter/100, log_file='aspirin_ani.csv', log_interval=1)

# %%
from torchmd.utils import LogWriter

logger = LogWriter(path="logs/", keys=('iter','ns','epot','ekin','etot','T'), name='aspirin_ani.csv')


# ----- Torchani --------
from tqdm import tqdm 
import numpy as np

FS2NS = 1E-6 # Femtosecond to nanosecond conversion

steps = 1000
output_period = 100
save_period = 100
traj = []

trajectoryout = "mytrajectory.npy"

iterator = tqdm(range(1, int(steps / output_period) + 1))
# system_ani.compute_forces()
# Epot = system.energy
for i in iterator:
    Ekin, Epot, T = integrator_ani.step(niter=output_period)
    # wrapper.wrap(system.pos, system.box)
    currpos = system_ani.pos.detach().cpu().numpy().copy()
    traj.append(currpos)
    
    if (i*output_period) % save_period  == 0:
        np.save(trajectoryout, np.stack(traj, axis=2))

    logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':(Epot+Ekin),'T':T})
    epot = system_ani.get_potential_energy() / len(system_ani)
    ekin = system_ani.get_kinetic_energy() / len(system_ani)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * kB), epot + ekin))

# %%
atom_types = aspirin_ase.get_chemical_symbols()

with open('aspirin_ani.xyz', 'w') as f:
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
view_trajectory(read('aspirin_ani.xyz', index=':'))
# %%
