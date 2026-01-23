import numpy as np

s = np.load("grassy_dit/data/moses/scattering_moments.npy")

J = 4  # J parameter
num_moments = 4

# This is the correct formula from _validate_inputs
num_levels = 1 + J + J*(J-1)//2  # = 1 + 11 + 55 = 67

scattering_dim = s.shape[1]
num_atom_types = scattering_dim // (num_levels * num_moments)

print(f"Shape: {s.shape}")
print(f"Scattering dim: {scattering_dim}")
print(f"J: {J}")
print(f"num_levels: {num_levels}")
print(f"num_moments: {num_moments}")
print(f"Atom types: {num_atom_types}")
print(f"Verification: {num_atom_types} * {num_levels} * {num_moments} = {num_atom_types * num_levels * num_moments}")