# %%
import ase
import torchani
import torch
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory

import time

calc = torchani.models.ANI1ccx(periodic_table_index=True).ase() # ASE


from ase.io import read
aspirin = ase.Atoms(read('alanine_dipeptide.xyz'))
aspirin.set_calculator(calc)

print("Begin minimizing...")
opt = LBFGS(aspirin)
opt.run(fmax=0.001)
print()


def printenergy(a=aspirin):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

dyn = Langevin(aspirin, 1 * units.fs, 300 * units.kB, 0.2, fixcm=False)
dyn.attach(printenergy, interval=100)

traj = Trajectory('alainine_ase.traj', 'w', aspirin)

dyn.attach(traj.write, interval=100)

print("Beginning dynamics...")
printenergy()
start_time = time.time()
n_iter = 2000
dyn.run(n_iter)
run_time = time.time() - start_time
print(f'Simulation took {run_time:.2f} seconds ({(n_iter/run_time):.2f} iter/sec)')
# %%
