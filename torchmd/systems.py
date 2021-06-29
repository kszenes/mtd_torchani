import torch
import numpy as np
import torchani
import ase.units
import nglview as nv

# BOLTZMAN = 0.001987191
class System_ANI:
    def __init__(self, model, pos, species, masses, natoms, symbols, nreplicas, precision, device):
        # self.pos = pos  # Nsystems,Natoms,3
        # self.vel = vel  # Nsystems,Natoms,3
        # self.box = box
        # self.forces = forces
        #self.box = torch.zeros(nreplicas, 3, 3)
        #self.pos = torch.zeros(nreplicas, natoms, 3)

        self.vel = torch.zeros(nreplicas, natoms, 3, device=device)
        self.forces = torch.zeros(nreplicas, natoms, 3, device=device)

        self.model = model.to(device)
        self.pos = pos.type(precision).to(device).requires_grad_(True)
        self.species = species.to(device)
        self.masses = masses.type(precision).to(device)
        self.symbols = symbols

        #self.precision_(precision)
        #self.to_(device)

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
        self.pos = self.pos.to(device)
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
    def from_ase(cls, ase_struct, device='cuda', precision='torch.float32'):
        """Initializes a System_ANI object from an ase Atoms object"""

        x = cls(torchani.models.ANI1ccx(periodic_table_index=True), torch.from_numpy(ase_struct.get_positions()).reshape(1, -1, 3),
        torch.tensor(ase_struct.get_atomic_numbers()).reshape(1, -1), torch.tensor(ase_struct.get_masses()).reshape(-1, 1),
        len(ase_struct), ase_struct.get_chemical_symbols(), 1, torch.float, device)

        return x

    def to_ase(self):
        """Returns an ase Atoms object corresponding to the Sytem_ANI"""
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

    def compute_forces(self):
        '''Returns forces in eV'''

    #    self.pos.requires_grad_(True)
    #    energy = torchani.units.hartree2ev(self.model((self.species, self.pos)).energies)
    #    self.energy = energy
    #    forces = -torch.autograd.grad(energy.sum(), self.pos)[0]
    #    self.forces = forces
    #    self.pos.requires_grad_(False)
        
        #Works but copies
        pos = self.pos.requires_grad_(True)
        energy = torchani.units.hartree2ev(self.model((self.species, pos)).energies)
        self.energy = energy
        forces = -torch.autograd.grad(energy.sum(), pos)[0]
        self.forces = forces

    def set_species(self, species):
        self.species = species

    def set_masses(self, masses):
        self.masses = masses

    def set_model(self, model):
        self.model = model

    def show_struct(self, gui=False):
        '''View molecule using Nglview'''
        x = self.to_ase()
        return nv.show_ase(x, gui=gui)

    # def get_dihedrals(self):
    #   '''Compute psi and phi using ase'''
    #     # only used for psi phi angles in ramachandran
    #     psi_list = [6, 8, 14, 16]
    #     phi_list = [4, 6, 8, 14]

    #     x = self.to_ase()
    #     dihedrals = x.get_dihedrals([psi_list, phi_list])
    #     dihedrals[dihedrals > 180] = dihedrals[dihedrals > 180] - 360 # Set between [-180, 180]
    #     return dihedrals

    def get_dihedral_ani(self, v0, v1, v2, degree=False):
        '''Computes dihedral between v0-v1 plane and v1-v2 plane natively in Pytorch''' 

        v1n = v1 / torch.norm(v1)
        v = -v0 - torch.einsum('ij, ij, ik -> ik', -v0, v1n, v1n)
        w = v2 - torch.einsum('ij, ij, ik -> ik', v2, v1n, v1n)

        x = torch.einsum('ij, ij -> i', v, w)
        y = torch.einsum('ij, ij -> i', torch.cross(v1n, v, axis=1), w)
        dihedral = torch.atan2(y, x)
        if degree == True:
            dihedral *= 180 / np.pi
        return dihedral

    def get_phi(self):
       '''Returns alanine dipeptide dihedral angle phi'''
       phi_list = [4, 6, 8, 14]
       x = self.pos[:, phi_list, :].requires_grad_(True)
       vec = x[:, 1:, :] - x[:, :3, :]
       return self.get_dihedral_ani(vec[:, 0, :], vec[:, 1, :], vec[:, 2, :])

    def get_psi(self):
       '''Return dihedrals angle psi'''
       psi_list = [6, 8, 14, 16]
       x = self.pos[:, psi_list, :].requires_grad_(True)
       vec = x[:, 1:, :] - x[:, :3, :]
       return self.get_dihedral_ani(vec[:, 0, :], vec[:, 1, :], vec[:, 2, :])

    def get_dihedrals_ani(self):
       '''Return dihedrals angle psi and phi'''
       psi_list = [6, 8, 14, 16]
       phi_list = [4, 6, 8, 14]

       x1 = self.pos[:, psi_list, :]
       x2 = self.pos[:, phi_list, :]

       vec1 = x1[:, 1:, :] - x1[:, :3, :]
       vec2 = x2[:, 1:, :] - x2[:, :3, :]        
       return torch.stack((self.get_dihedral_ani(vec1[:, 0, :], vec1[:, 1, :], vec1[:, 2,:]), self.get_dihedral_ani(vec2[:, 0, :], vec2[:, 1, :], vec2[:, 2,:])), axis=1)
    #    return torch.tensor([self.get_dihedral_ani(vec1[:, 0, :], vec1[:, 1, :], vec1[:, 2,:]), self.get_dihedral_ani(vec2[:, 0, :], vec2[:, 1, :], vec2[:, 2,:])], requires_grad=True)





