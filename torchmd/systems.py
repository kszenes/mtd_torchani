import torch
import numpy as np


class System:
    def __init__(self, natoms, nreplicas, precision, device):
        # self.pos = pos  # Nsystems,Natoms,3
        # self.vel = vel  # Nsystems,Natoms,3
        # self.box = box
        # self.forces = forces
        self.box = torch.zeros(nreplicas, 3, 3)
        self.pos = torch.zeros(nreplicas, natoms, 3)
        self.vel = torch.zeros(nreplicas, natoms, 3)
        self.forces = torch.zeros(nreplicas, natoms, 3)

        self.to_(device)
        self.precision_(precision)

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nreplicas(self):
        return self.pos.shape[0]

    def to_(self, device):
        self.forces = self.forces.to(device)
        self.box = self.box.to(device)
        self.pos = self.pos.to(device)
        self.vel = self.vel.to(device)

    def precision_(self, precision):
        self.forces = self.forces.type(precision)
        self.box = self.box.type(precision)
        self.pos = self.pos.type(precision)
        self.vel = self.vel.type(precision)

    def set_positions(self, pos):
        if pos.shape[1] != 3:
            raise RuntimeError(
                "Positions shape must be (natoms, 3, 1) or (natoms, 3, nreplicas)"
            )

        atom_pos = np.transpose(pos, (2, 0, 1))
        if self.nreplicas > 1 and atom_pos.shape[0] != self.nreplicas:
            atom_pos = np.repeat(atom_pos[0][None, :], self.nreplicas, axis=0)

        self.pos[:] = torch.tensor(
            atom_pos, dtype=self.pos.dtype, device=self.pos.device
        )

    def set_velocities(self, vel):
        if vel.shape != (self.nreplicas, self.natoms, 3):
            raise RuntimeError("Velocities shape must be (nreplicas, natoms, 3)")
        self.vel[:] = vel.clone().detach().type(self.vel.dtype).to(self.vel.device)

    def set_box(self, box):
        if box.ndim == 1:
            if len(box) != 3:
                raise RuntimeError("Box must have at least 3 elements")
            box = box[:, None]

        if box.shape[0] != 3:
            raise RuntimeError("Box shape must be (3, 1) or (3, nreplicas)")

        box = np.swapaxes(box, 1, 0)

        if self.nreplicas > 1 and box.shape[0] != self.nreplicas:
            box = np.repeat(box[0][None, :], self.nreplicas, axis=0)

        for r in range(box.shape[0]):
            self.box[r][torch.eye(3).bool()] = torch.tensor(
                box[r], dtype=self.box.dtype, device=self.box.device
            )

    def set_forces(self, forces):
        if forces.shape != (self.nreplicas, self.natoms, 3):
            raise RuntimeError("Forces shape must be (nreplicas, natoms, 3)")
        self.forces[:] = torch.tensor(
            forces, dtype=self.forces.dtype, device=self.forces.device
        )

import torchani
import ase.units
import nglview as nv

BOLTZMAN = 0.001987191
class System_ANI:
    def __init__(self, model, pos, species, masses, natoms, symbols, nreplicas, precision, device):
        # self.pos = pos  # Nsystems,Natoms,3
        # self.vel = vel  # Nsystems,Natoms,3
        # self.box = box
        # self.forces = forces
        self.box = torch.zeros(nreplicas, 3, 3)
        self.pos = torch.zeros(nreplicas, natoms, 3)
        self.vel = torch.zeros(nreplicas, natoms, 3)
        self.forces = torch.zeros(nreplicas, natoms, 3)

        self.model = model
        self.pos = pos
        self.species = species
        self.masses = masses
        self.symbols = symbols

        self.precision_(precision)
        self.to_(device)

        self.compute_forces()

    def __len__(self):
        return self.masses.shape[0]

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nreplicas(self):
        return self.pos.shape[0]

    def to_(self, device):
        self.forces = self.forces.to(device)
        self.box = self.box.to(device)
        self.pos = self.pos.requires_grad_(True).to(device)
        self.vel = self.vel.to(device)

        self.model = self.model.to(device)
        self.species = self.species.to(device)
        self.masses = self.masses.to(device)

    def precision_(self, precision):
        self.forces = self.forces.type(precision)
        self.box = self.box.type(precision)
        self.pos = self.pos.type(precision)
        self.vel = self.vel.type(precision)

    @classmethod
    def from_ase(cls, ase_struct, device='cpu', precision='torch.float'):
        x = cls(torchani.models.ANI1ccx(periodic_table_index=True), torch.from_numpy(ase_struct.get_positions()).reshape(1, -1, 3),
        torch.tensor(ase_struct.get_atomic_numbers()).reshape(1, -1), torch.tensor(ase_struct.get_masses()).reshape(-1, 1),
        len(ase_struct), ase_struct.get_chemical_symbols(), 1, torch.float, device)

        return x

        # self.model = torchani.models.ANI1ccx(periodic_table_index=True)
        # self.species = torch.tensor(ase_struct.get_atomic_numbers).reshape(-1, 5)
        # self.pos = torch.tensor(ase_struct.get_positions()).reshape(1, -1, 3)
        # self.masses = torch.tensor(ase_struct.get_masses()).reshape(-1, 1)

        # self.to_(device)
        # self.precision_(precision)

    def to_ase(self):
        x = ase.Atoms(self.get_species().cpu().reshape(-1), self.get_positions().detach().cpu().reshape(-1, 3))
        return x

    def get_species(self):
        return self.species

    def get_symbols(self):
        return self.symbols

    def get_masses(self):
        return self.masses

    def get_center_of_mass(self):
        masses = self.get_masses()
        com = masses * self.get_positions() / masses.sum()
        return com

    def get_positions(self):
        return self.pos

    def set_positions(self, pos):
        # if pos.shape[1] != 3:
        #     raise RuntimeError(
        #         "Positions shape must be (natoms, 3, 1) or (natoms, 3, nreplicas)"
        #     )

        # atom_pos = np.transpose(pos, (2, 0, 1))
        # if self.nreplicas > 1 and atom_pos.shape[0] != self.nreplicas:
        #     atom_pos = np.repeat(atom_pos[0][None, :], self.nreplicas, axis=0)

        # self.pos[:] = torch.tensor(
        #     atom_pos, dtype=self.pos.dtype, device=self.pos.device
        # )
        self.pos = pos.float()

    def get_velocities(self):
        return self.vel

    def set_velocities(self, vel):
        if vel.shape != (self.nreplicas, self.natoms, 3):
            raise RuntimeError("Velocities shape must be (nreplicas, natoms, 3)")
        self.vel[:] = vel.clone().detach().type(self.vel.dtype).to(self.vel.device)

    def get_kinetic_energy(self):
        Ekin = torch.sum(0.5 * torch.sum(self.get_velocities() * self.get_velocities(), dim=2, keepdim=True) * self.get_masses(), dim=1)
        return Ekin.item()

    def get_potential_energy(self):
        return self.energy.item()

    def get_total_energy(self):
        return self.get_potential_energy() + self.get_kinetic_energy()

    def get_forces(self):
        self.compute_forces()
        return self.forces

    def get_temperature(self):
        dof = len(self) * 3
        return 2 * self.get_kinetic_energy() / (dof * ase.units.kB)
        # return 2 * self.get_kinetic_energy() / (dof * BOLTZMAN)

    def set_box(self, box):
        if box.ndim == 1:
            if len(box) != 3:
                raise RuntimeError("Box must have at least 3 elements")
            box = box[:, None]

        if box.shape[0] != 3:
            raise RuntimeError("Box shape must be (3, 1) or (3, nreplicas)")

        box = np.swapaxes(box, 1, 0)

        if self.nreplicas > 1 and box.shape[0] != self.nreplicas:
            box = np.repeat(box[0][None, :], self.nreplicas, axis=0)

        for r in range(box.shape[0]):
            self.box[r][torch.eye(3).bool()] = torch.tensor(
                box[r], dtype=self.box.dtype, device=self.box.device
            )

    def compute_forces(self):

        energy = torchani.units.hartree2ev(self.model((self.species, self.pos)).energies)
        self.energy = energy
        forces = -torch.autograd.grad(energy.sum(), self.pos)[0]
        self.forces = forces
        

        # Works but copies
        # pos = self.pos.float().requires_grad_(True)
        # energy = torchani.units.hartree2ev(self.model((self.species, pos)).energies)
        # self.energy = energy
        # forces = -torch.autograd.grad(energy.sum(), pos)[0]
        # self.forces = forces

    def set_species(self, species):
        self.species = species

    def set_masses(self, masses):
        self.masses = masses

    def set_model(self, model):
        self.model = model

    def show_struct(self, gui=False):
        x = self.to_ase()
        return nv.show_ase(x, gui=gui)

    def get_dihedrals(self):
        # only used for psi phi angles in ramachandran
        psi = [7, 6, 8, 10] # Nitrogen
        phi = [15, 14, 8, 10] # Oxygen

        x = self.to_ase()
        dihedrals = x.get_dihedrals([phi, psi])
        dihedrals[dihedrals > 180] = dihedrals[dihedrals > 180] - 360 # Set between [-180, 180]
        return dihedrals
