import numpy as np
import torch
from torchani.units import hartree2ev
import ase.units

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191



def kinetic_energy(masses, vel):
    Ekin = torch.sum(0.5 * torch.sum(vel * vel, dim=2, keepdim=True) * masses, dim=1)
    return Ekin


def maxwell_boltzmann(masses, T, replicas=1, device='cuda'):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            # torch.sqrt(T * BOLTZMAN / masses) * torch.randn((natoms, 3)).type_as(masses)
            torch.sqrt(T * ase.units.kB / masses) * torch.randn((natoms, 3), device=device)
        )

    return torch.stack(velocities, dim=0)


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel

def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, gamma, coeff, dt, device):
    csi = torch.randn_like(vel, device=device) * coeff
    vel += -gamma * vel * dt + csi

def _first_VV_ani(system, dt):
    accel = system.forces / system.masses
    # pos += vel * dt + 0.5 * accel * dt * dt

    # pos = system.pos.detach()
    # pos += system.vel * dt + 0.5 * accel * dt * dt
    # system.pos = torch.tensor(pos, requires_grad=True)
    with torch.no_grad():
        system.pos += system.vel * dt + 0.5 * accel * dt * dt

    # system.pos = (system.pos + system.vel * dt + 0.5 * accel * dt * dt).float()
    system.vel += 0.5 * dt * accel

def _second_VV_ani(system, dt):
    system.compute_forces()
    accel = system.forces / system.masses
    system.vel += 0.5 * dt * accel


def langevin_ani(system, gamma, coeff, dt, device):
    csi = torch.randn_like(system.vel, device=device) * coeff
    system.vel += -gamma * system.vel * dt + csi



PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


class Integrator:
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None):
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
        self.device = device
        gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T:
            M = self.forces.par.masses
            self.vcoeff = torch.sqrt(2.0 * gamma / M * BOLTZMAN * T * self.dt).to(
                device
            )

    def step(self, niter=1):
        s = self.systems
        masses = self.forces.par.masses
        natoms = len(masses)
        for _ in range(niter):
            _first_VV(s.pos, s.vel, s.forces, masses, self.dt)
            pot = self.forces.compute(s.pos, s.box, s.forces)
            if self.T:
                langevin(s.vel, self.gamma, self.vcoeff, self.dt, self.device)
            _second_VV(s.vel, s.forces, masses, self.dt)

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T = kinetic_to_temp(Ekin, natoms)
        return Ekin, pot, T

class Integrator_ANI:
    def __init__(self, systems, timestep, device, gamma=None, T=None):
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = systems.forces
        self.device = device
        self.masses = systems.masses
        gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T:
            M = self.masses
            self.vcoeff = torch.sqrt(2.0 * gamma / M * BOLTZMAN * T * self.dt).to(
                device
            )

    def step(self, niter=1):
        s = self.systems
        masses = self.masses
        natoms = len(masses)
        for _ in range(niter):
            _first_VV_ani(s, self.dt)
            # pot = self.forces.compute(s.pos, s.box, s.forces)
            if self.T:
                langevin_ani(s, self.gamma, self.vcoeff, self.dt, self.device)
            _second_VV_ani(s, self.dt)

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T = kinetic_to_temp(Ekin, natoms)
        s.compute_forces()

        return Ekin, s.energy, T


class Langevin_integrator:
    def __init__(self, system, dt, device, fr=None, temp=None, fix_com=True, height=0.004336, width=0.05):
        self.fr = fr 
        self.temp = temp * ase.units.kB
        self.dt = dt * ase.units.fs
        self.system = system
        self.masses = self.system.masses
        self.fix_com = fix_com
        self.initial_height = height
        self.height = torch.tensor([height], device=device)

        self.width = width

        self.updatevars()

    def updatevars(self):
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
        # start_time = time.time()
        if traj_file is not None:
            if append == False:
                traj_f = open(traj_file, 'w')
            else:
                traj_f = open(traj_file, 'a')

        if log_file is not None and log_file != '-':
            if append == False:
                log_f = open(log_file, 'w')
            else:
                log_f = open(log_file, 'a')

            if per_atom:
                log_f.write('# per atom \n')
            log_f.write('Epot,' + 'Ekin,' + 'Etot,' + 'Phi,Psi,' + 'Temp\n') # Ramachandran
            # log_f.write('Epot,' + 'Ekin,' + 'Etot,' + 'Temp\n')


        if metadyn is not None and append == False:
            self.peaks = self.get_cv().detach()
            self.n_cv = torch.numel(self.peaks)
            if metadyn == 'well-tempered':
                if dTemp is None:
                    raise Exception('dTemp must be provided for well-tempered metadynamics')



        for i in tqdm(range(n_iter)):
            if traj_file is not None and i % traj_interval == 0:
                self.write_traj(traj_f)

            if log_file is not None and i % log_interval == 0:
                if log_file == 'std_out':
                    self.print_log(per_atom)
                else:
                    self.write_log(log_f, per_atom)

            self.step(metadyn=metadyn, device=device)
            if metadyn is not None and i % 20 == 19:
                bias = self.get_bias(self.get_cv())
                if dTemp is not None:
                    self.height = torch.cat((self.height, torch.tensor([self.initial_height * torch.exp(- bias / (ase.units.kB * dTemp))], device=device)))
                self.peaks = torch.cat((self.peaks, self.get_cv().detach()))
                # if self.n_cv > 1:
                #     self.peaks = torch.cat((self.peaks, (self.get_cv().detach()))) # high-dim potential
                # else:
                    # self.peaks = torch.cat((self.peaks, (self.get_cv().detach()))) # 1d potential


        if traj_file is not None:
            traj_f.close()

        if log_file is not None and log_file != '-':
            log_f.close()

        # run_time = time.time() - start_time
        # print(f"----------- Simulation took {(run_time):.2f} sec ({(n_iter/run_time):.2f} iters/sec) -----------")

    def get_cv(self):
        # return self.system.get_phi()
        return self.system.get_dihedrals_ani()

    def get_bias(self, cv):
        ''' 
        https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/metadynamics
        height = 0.1 kcal/mol = 0.004336 eV
        width = 0.05
        deposition rate = 2ps
        sampling time = 500ns

        Plumed masterclass 21-4.2
        initial height = 1.2 kcal/mol = 0.052 eV
        bias factor = 8
        '''
        if self.n_cv > 1:
            # print(torch.sum(self.height * torch.exp(-(cv[:, 0] - self.peaks[:, 0])**2 / (2*self.width**2)) * torch.exp(-(cv[:, 1] - self.peaks[:, 1])**2 / (2*self.width**2))))
            return torch.sum(self.height * torch.prod(torch.exp(- (cv - self.peaks)**2 / (2 * self.width**2)), dim=1))
        else:
            return torch.sum(self.height * torch.exp(- (cv - self.peaks)**2 / (2 * self.width**2)))


    def get_bias_forces(self):
        bias = self.get_bias(self.get_cv())
        f_bias = -torch.autograd.grad(bias, self.system.pos)[0]
        return f_bias

    def get_gauss_height(self, cv, well_tempered=False):
        if not well_tempered:
            return self.height

    def get_free_energy(self, n_points=1000):
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
