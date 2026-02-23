[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/dindagustiayu/The-moment-of-Inertia-of-H2O/blob/main/Moment%20inertia%20of%20molecule.cpp)


# The moment inertia of a molecule

## Moment inertia
The moment of inertia of a molecule may be defined as the product of mass of each atom and the square of its distance from the rotational axis through the centre of mass of the molecule. Mathematically, it may be written as,
<p align='center'>
    $$I = \sum_{i}\;m_{i}r_{i}^{2}$$
</p>

where $r_{i}$ is the distance of each atom from the centre of mass.

The moment of inertia of a molecule may be resolved into rotational components about three mutually perpendicular directions through the centre of gravity. The quantity $I$, which translates one vector into another vector, is therefore a __matrix__, called the __inertia matrix__:

<p align='center'>
    $$I=\begin{pmatrix}I_{xx} & I_{xy} & -I_{xz}\\ I_{yx} & I_{yy} & I_{yz}\\ I_{zx} & I_{zy} & I_{zz} \end{pmatrix}$$
</p>

where the __diagonal__ elements (called "moment of inertia") are similar to the one-dimensional definition, e.g.

<p align='center'>
    $$I_{xx}=\sum_{i}m_{i}(y_{i}^{2}+z_{i}^{2}),$$
    $$I_{yy}=\sum_{i}m_{i}(x_{i}^{2}+z_{i}^{2}),$$
    $$I_{zz}=\sum_{i}m_{i}(x_{i}^{2}+y_{i}^{2});$$
</p>

but the __off-diagonal__ elements (called "products of inertia") must also be included:

<p align='center'>
    $$I_{xy}= I_{yx} = -\sum_{i}m_{i}x_{i}y_{i},$$
    $$I_{xz}= I_{zx} = -\sum_{i}m_{i}x_{i}z_{i},$$
    $$I_{yz}= I_{zy} = -\sum_{i}m_{i}y_{i}z_{i}.$$
</p>

In the case of a rotation molecule, the index __i__ runs over the atoms in the molecule.
Thus, a molecule has three principal moments of inertia, usually designated as $I_{A}, I_{B}, I_{C}$. The three principal moments of inertia may be taken as,
- $I_{A}$ for rotation about the bond axis
- $I_{B}$ for end-over-end rotation in the plane of the paper
- $I_{C}$ for end-over-end rotation at right angles to the plane of the paper. 

Based on the values of $I_{A},\;I_{B}, and\;I_{C}$, molecules may be classified into several group as:

- $I_{A}=0$ while $I_{B}=I_{C}$ : __Linear molecule__, examples: $CO_{2}$, HCl, etc.
  
- $I_{B}=I_{C}\neq I_{A}$, while $I_{A}\neq 0$ : __Symmetric top molecule__.
  - (a) if $I_{B}=I_{C} >  I_{A}$ : __Prolate symmetric top molecule__. eg. $CH_{3}Cl$.
  - (b) if $I_{B}=I_{C} <  I_{A}$ : __Oblate symmetric top molecule__. eg. $BCl_{3}$.

- $I_{A}=I_{B}=I_{C}$ : __Spherical top molecule__. eg. $CH_{4}$

- $I_{A}\neq I_{B}\neq I_{C}$ : __Asymmetric top molecule__.$CHCl$.

Furthermore, in spectroscopy, it is conventional to define the _rotational constants_,

<p align='center'>
    $$A=\frac{h}{8\pi^{2}cI_{A}},$$
    $$A=\frac{h}{8\pi^{2}cI_{B}},$$
    $$A=\frac{h}{8\pi^{2}cI_{C}},$$
</p>

which are reported in wavenumber units ($cm^{-1}$).

The file [H2O.dat](https://github.com/dindagustiayu/The-moment-of-Inertia-of-H2O/blob/main/H2O.dat) contains the positions of the atoms in the molecule H2O in XYZ format. Determine the rotational constants for this molecule and classify it as a spherical, oblate, prolate, or asymmetric top.

```python
# H2O
# mass		x           	y           	z       
  15.999	0.00000000  	0.00000000  	0.11779		#O
  1.00784	-0.53418382 	0.53418382 	    -0.47116 	#H
  1.00784	0.53418382 	    -0.53418382 	-0.47116 	#H
```

The four columns of the provided data file are mass (in Da) and (x, y, z) coordinates (in $\mathring{A}$).

```python
import numpy as np
from scipy.constants import u, h, c
m, x, y, z = np.genfromtxt('H2O.dat', unpack=True)
```

To ensure the atomic coordinates are stored relative to the molecular center of mass, we must shift their origin to this position. In the provided coordinates, thecenter of mass is at.

<p align='center'>
  $$rCM=\frac{1}{M} \sum_{i} m_{i}r_{i},$$ where $$M=\sum_{i} m_{i}$$
</p>


```python
def translate_to_cofm(m, x, y, z):
    """ Translate the atom positions to be relative to the CofM."""

    # Total molecular mass.
    M = np.sum(m)

    # position of center of mass in original coordinates.
    xCM = np.sum(m * x) / M
    yCM = np.sum(m * y) / M
    zCM = np.sum(m * z) / M

    # Transform to CoFM coordinates and return them
    return x - xCM, y - yCM, z - zCM

def get_inertia_matrix(m, x, y, z):
    """ Return the moment of inertia tensor."""

    x, y, z = translate_to_cofm(m, x, y, z)
    Ixx = np.sum(m * (y**2 + z**2))
    Iyy = np.sum(m * (x**2 + z**2))
    Izz = np.sum(m * (x**2 + y**2))
    Ixy = -np.sum(m * x * y)
    Iyz = -np.sum(m * y * z)
    Ixz = -np.sum(m * x * z)
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]
                 ])
    return I

```

This function is used in the code below to construct and diagonalize the moment of inertia matrix.

```python
def get_principal_moi(I):
    """ Determine the principal moments of inertia."""

    # The principal moments of inertia are the eigenvalues of the moment of inertia tensor.
    Ip = np.linalg.eigvals(I)

    # Sort and convert principal moments of inertia to kg.m2 before returning.
    Ip.sort()
    return Ip * u / 1e20

def get_rotational_constants(filename):
    """ Return the rotational constants, A, B, C (in cm-1) for a molecule.

    The atomic coordinates are retrieved from the filename which should have
    Four columns of data: mass (in Da), and x, y, z coordinates (in Angstroms).

    """
    m, x, y, z = np.genfromtxt(filename, unpack=True)
    I = get_inertia_matrix(m, x, y, z)
    Ip = get_principal_moi(I)
    A, B, C = h / 8 / np.pi**2 / c / 100 / Ip
    return A, B, C

A, B, C = get_rotational_constants('H2O.dat')
print(f'H2O: A = {A} cm-1, B = {B} cm-1, C = {C} cm-1')
```
```
H2O: A = 27.148870802103488 cm-1, B = 14.654245226373835 cm-1, C = 9.51714245613036 cm-1
```
We have $A\neq B\neq C$, it must be that $I_{A} \neq I_{B} \neq I_{C}$, and $H_{2}O$ is __Asymmetric top__.These values are consistent with published spectroscopy data. [J. Chem. Phys. 24, 1139–1165 (1956)](https://doi.org/10.1063/1.1742731).
