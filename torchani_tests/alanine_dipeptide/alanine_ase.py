# %%
import ase
import torchani
import torch
from ase.optimize import LBFGS
from ase.md.langevin import Langevin
from ase import units
from ase.io.trajectory import Trajectory
import sys

import time

calc = torchani.models.ANI1ccx(periodic_table_index=True).ase() # ASE

structure = sys.argv[1]


from ase.io import read
alanine = ase.Atoms(read(structure))
alanine.set_calculator(calc)

print("Begin minimizing...")
opt = LBFGS(alanine)
opt.run(fmax=0.001)
print()

psi_list = [7, 6, 8, 10] # Nitrogen
phi_list = [15, 14, 8, 10] # Oxygen


with open('./log/alaning_ase_B.csv', 'w') as f:

    def printenergy(a=alanine, log_file=f):
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        etot = epot + ekin
        psi, phi = a.get_dihedrals([psi_list, phi_list])
        temp = a.get_temperature()
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
            'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
        log_file.write(str(epot) +  ',' + str(ekin) + ',' + str(etot) + ',' + str(phi) + ',' + str(psi) + ',' + str(temp) + '\n')

    dyn = Langevin(alanine, 1 * units.fs, 300 * units.kB, 0.2, fixcm=False)
    dyn.attach(printenergy, interval=1)

    # traj = Trajectory('alainine_ase.traj', 'w', alanine)

    # dyn.attach(traj.write, interval=100)
    f.write('Epot,' + 'Ekin,' + 'Etot,' + 'Phi,Psi,' + 'Temp\n') # Ramachandran
    print("Beginning dynamics...")
    printenergy()
    start_time = time.time()
    n_iter = int(1e4)
    dyn.run(n_iter)
    run_time = time.time() - start_time
    print(f'Simulation took {run_time:.2f} seconds ({(n_iter/run_time):.2f} iter/sec)')
# %%
