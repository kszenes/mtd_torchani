import numpy as np
import torch
from torchani.units import hartree2ev
import ase.units

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# TIMEFACTOR = 48.88821
# BOLTZMAN = 0.001987191
# PICOSEC2TIMEU = 1000.0 / TIMEFACTOR



def maxwell_boltzmann(masses, T, replicas=1, device='cuda'):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            # torch.sqrt(T * BOLTZMAN / masses) * torch.randn((natoms, 3)).type_as(masses)
            torch.sqrt(T * ase.units.kB / masses) * torch.randn((natoms, 3), device=device)
        )

    return torch.stack(velocities, dim=0)


class Langevin_integrator:
    '''Langevin integrator ported from ase. Based on paper: 10.1103/PhysRevE.75.056707

        Parameters recommended to run simulation:

        https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/metadynamics
        height = 0.1 kcal/mol = 0.004336 eV
        width = 0.05
        deposition rate = 2ps
        sampling time = 500ns

        Plumed masterclass 21-4.2
        initial height = 1.2 kcal/mol = 0.052 eV
        bias factor = (temp + dTemp) / temp = 8
    '''

    def __init__(self, system, dt, device='cuda', fr=None, temp=None, fix_com=True, height=0.004336, width=0.05):
        '''
        Parameters
        ----------
        system : System_ANI
        dt : int
            Timestep (fs)
        device : torch.device
        fr : float
            Friction coefficient
        temp : int
            Temperature
        fix_com : bool
            Recenter system after each step
        height : float
            Height of gaussian for metadynamics
        width : float
            Width of gaussian for metadynamics
        '''

        self.fr = fr 
        self.temp = temp * ase.units.kB
        self.dt = dt * ase.units.fs
        self.system = system
        self.masses = self.system.masses
        self.fix_com = fix_com
        self.initial_height = height
        self.height = torch.tensor([height], device=device)
        self.width = width
        self.device = device

        self.updatevars()

    def updatevars(self):
        '''Update coefficients used in Langevin integrator'''
        dt = self.dt
        T = self.temp 
        fr = self.fr
        masses = self.masses
        sigma = torch.sqrt(2 * T * fr / masses)

        self.c1 = dt / 2. - dt * dt * fr / 8.
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.
        self.c3 = np.sqrt(dt) * sigma / 2. - dt**1.5 * fr * sigma / 8.
        self.c5 = dt**1.5 * sigma / (2 * np.sqrt(3))
        self.c4 = fr / 2. * self.c5

    def step(self, forces=None, metadyn=None, device='cuda'):
        '''Do a step of integration
        
        Parameters
        ----------
        forces : not implemented
            Add external force
        metadym : None (MD), True (Metadynamics), well-tempered (Well-tempered metadynamics)
            Type of simulation to run
        '''

        system = self.system
        n_system = len(system)

        if forces is None:
            forces = system.get_forces()

        if metadyn is not None:
            forces += self.get_bias_forces()

        self.vel = system.get_velocities()

        self.xi = torch.randn(size=(n_system, 3), device=device)
        self.eta = torch.randn(size=(n_system, 3), device=device)

        self.vel += (self.c1 * forces / self.masses - self.c2 * self.vel +
                self.c3 * self.xi - self.c4 * self.eta)

        x = system.get_positions().detach()
        system.set_positions(x + self.dt * self.vel + self.c5 * self.eta)

        self.vel = (self.system.get_positions() - x -
                self.c5 * self.eta) / self.dt
        forces = system.get_forces()

        self.vel += (self.c1 * forces / self.masses - self.c2 * self.vel +
                self.c3 * self.xi - self.c4 * self.eta)
        system.set_velocities(self.vel)
        # return system.get_kinetic_energy(), system.get_potential_energy(), system.get_temperature()

    def run(self, n_iter=1, traj_file=None, traj_interval=1, log_file=None, log_interval=1, per_atom=True, metadyn=None, metadyn_func=None, dTemp=None, append=False, device='cuda'):
        '''Run simulation

        Paremeters
        ----------
        n_iter : int
            Number of steps to do
        traj_file : None or string
            Name of file to output trajectory (xyz format)
        traj_interval : int
            Output trajectory every traj_interval timesteps
        log_file : None or string
            Name of file to output energies, temperature, dihedrals ...  (csv format)
        log_interval : int
            Output csv every log_interval timesteps
        per_atom : bool
            Output log information per atom
        metadyn : None (MD), True (Metadynamics), well-tempered (Well-tempered metadynamics)
            Type of simulation to run
        metadyn_func : function
            Function that returns CV of system in a configuration
        dTemp : float
            Delta Temp used in well-tempered metadynamics
        append : bool
            If run interactively (e.g. Jupyter), continue running simulation from where it ended (does not appends to traj/log file as well as peaks)
        '''

        if traj_file is not None:
            if append == False:
                traj_f = open(traj_file, 'w')
            else:
                traj_f = open(traj_file, 'a')

        # log_file == '-' indicates to print to stdout
        if log_file is not None and log_file != '-':
            if append == False:
                log_f = open(log_file, 'w')
            else:
                log_f = open(log_file, 'a')

            if per_atom:
                log_f.write('# per atom \n')
            log_f.write('Epot,' + 'Ekin,' + 'Etot,' + 'Phi,Psi,' + 'Temp\n') # Ramachandran
            # log_f.write('Epot,' + 'Ekin,' + 'Etot,' + 'Temp\n')


        # Initialize metadynamics parameters
        if metadyn is not None and append == False:
            self.metadyn_func = metadyn_func
            self.peaks = self.get_cv().detach()  # peaks of gaussian
            self.n_cv = torch.numel(self.peaks)  # number of collective variables CV
            if metadyn == 'well-tempered':
                if dTemp is None:
                    raise Exception('dTemp must be provided for well-tempered metadynamics')
                else:
                    self.dTemp = dTemp



        # Tqdm provides visual progess bar
        for i in tqdm(range(n_iter)):
            if traj_file is not None and i % traj_interval == 0:
                self.write_traj(traj_f)

            if log_file is not None and i % log_interval == 0:
                if log_file == 'std_out':
                    self.print_log(per_atom)
                else:
                    self.write_log(log_f, per_atom)

            self.step(metadyn=metadyn, device=device)

            # Add gaussian every 20 fs
            if metadyn is not None and i % 20 == 19:
                bias = self.get_bias(self.get_cv())
                if metadyn == 'well-tempered':
                    # add height of gaussian for well-tempered metad
                    self.height = torch.cat((self.height, self.get_gauss_height()))

                # add location of peak of gaussianght()))
                self.peaks = torch.cat((self.peaks, self.get_cv().detach()))


        if traj_file is not None:
            traj_f.close()

        if log_file is not None and log_file != '-':
            log_f.close()

    def get_cv(self):
        '''Returns collective variable for metadynamics'''
        return self.metadyn_func()

    def get_bias(self, cv):
        '''Returns bias potential at a certain CV'''
        # Seperate 1 dim CV and higher dim CV
        if self.n_cv > 1:
            return torch.sum(self.height * torch.prod(torch.exp(- (cv - self.peaks)**2 / (2 * self.width**2)), dim=1))
        else:
            return torch.sum(self.height * torch.exp(- (cv - self.peaks)**2 / (2 * self.width**2)))


    def get_bias_forces(self):
        '''Returns forces from bias'''
        bias = self.get_bias(self.get_cv())
        f_bias = -torch.autograd.grad(bias, self.system.pos)[0]
        return f_bias

    def get_gauss_height(self):
        '''Returns height of gaussian for well-tempered metadynamics'''
        bias = self.get_bias(self.get_cv())
        return torch.tensor([self.initial_height * torch.exp(- bias / (ase.units.kB * self.dTemp))], device=self.device)

    def get_free_energy(self, n_points=1000):
        '''Not implemented yet'''
        phi_range = torch.arange(-np.pi, np.pi, 2*np.pi/n_points)
        psi_range = torch.arange(-np.pi, np.pi, 2*np.pi/n_points)
        # x_range = torch.arange(-np.pi, np.pi, 2*np.pi/n_points)
        discretisation = torch.stack((psi_range, phi_range), 0)
        
        gauss = - self.height * torch.sum(torch.exp(-(x_range - self.peaks[:,None])**2 / (2*self.width**2)), dim=0)
        plt.plot(x_range, gauss)
        plt.show()
        return torch.stack((x_range, gauss), dim=1)



    def write_traj(self, traj_file):
        atom_types = self.system.get_symbols()
        # TODO check why pos needs detach() !
        coord = self.system.get_positions().detach().view(-1, 3)
        traj_file.write(str(coord.shape[0]) + '\n\n')
        for j in range(coord.shape[0]):
            traj_file.write(atom_types[j])
            for k in range(3):
                traj_file.write(' ' + str(coord[j, k].item())) 
            traj_file.write('\n')

    def write_log(self, log_file, per_atom):
        n = 1
        if per_atom:
            n = len(self.system)

        epot = self.system.get_potential_energy() / n
        ekin = self.system.get_kinetic_energy() / n
        etot = self.system.get_total_energy() / n
        temp = self.system.get_temperature()
        # log_file.write(str(epot) +  ',' + str(ekin) + ',' + str(etot) + ',' + str(temp) + '\n')


        psi = self.system.get_dihedrals_ani()[0, 0]
        phi = self.system.get_dihedrals_ani()[0, 1]
        log_file.write(str(epot) +  ',' + str(ekin) + ',' + str(etot) + ',' + str(phi.item()) + ',' + str(psi.item()) + ',' + str(temp) + '\n')

    def print_log(self, per_atom):
        n = 1
        if per_atom:
            n = len(self.system)

        epot = self.system.get_potential_energy() / n
        ekin = self.system.get_kinetic_energy() / n
        etot = self.system.get_total_energy() / n
        temp = self.system.get_temperature()

        if per_atom:
            print(f'Energy per atom: Epot = {epot:.3f}eV  Ekin = {ekin:.3f}eV (T = {int(temp)}K) Etot = {etot:.3f}eV')
        else:
            print(f'Energy: Epot = {epot}eV  Ekin = {ekin}eV (T = {temp}K) Etot = {etot}eV')
