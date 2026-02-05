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
but the __off-diagobal__ elements (called "products of inertia") must also be included:
<p align='center'>
    $$I_{xy}= I_{yx} = \sum_{i}m_{i}x_{i}y_{i},$$
    $$I_{xz}= I_{zx} = \sum_{i}m_{i}x_{i}z_{i},$$
    $$I_{yz}= I_{zy} = \sum_{i}m_{i}y_{i}z_{i}.$$
</p>
In the case of a rotation molecule, the index $i$ runs over the atoms in the molecule.
Thus, a molecule has three principal moments of inertia, usually designated as $I_{A},\;I_{B},\;I_{C}$. The three principal moments of inertia may be taken as,
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

Furtheremore, in spectroscopy, it is conventional to define the _rotational constants_, 
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
The four column of the provided data file are mass (in Da) and (x, y, z) coordinates (in $\mathring{A}$).

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Dense>   // Eigen is used for matrix operations and eigenvalues

// ---------------------------------------------------------------------
// Physical constants (SI units)
// ---------------------------------------------------------------------
constexpr double atomic_mass_unit = 1.66053906660e-27;      // kg  (u)
constexpr double planck_constant   = 6.62607015e-34;       // J·s (h)
constexpr double speed_of_light    = 2.99792458e8;         // m/s (c)
constexpr double pi                = M_PI;                // π

// ---------------------------------------------------------------------
// Helper type aliases
// ---------------------------------------------------------------------
using Vector  = std::vector<double>;
using Matrix3 = Eigen::Matrix3d;

// ---------------------------------------------------------------------
// Read four‑column data file (mass, x, y, z)
// ---------------------------------------------------------------------
static bool readMoleculeData(const std::string& filename,
                             Vector& masses,
                             Vector& xs,
                             Vector& ys,
                             Vector& zs)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Error: cannot open file \"" << filename << "\"\n";
        return false;
    }

    double m, x, y, z;
    while (fin >> m >> x >> y >> z) {
        masses.push_back(m);
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
    }
    return true;
}
```

To ensure the atomic coordinates are stored relative to the molecular center of mass, we must shift their origin to this position. In the provided coordinates, thecenter of mass is at.

<p align='center'>
  $$rCM=\frac{1}{M} \sum_{i} m_{i}r_{i},$$ where $$M=\sum_{i} m_{i}$$
</p>


```cpp

// ---------------------------------------------------------------------
// Translate atomic coordinates to the centre‑of‑mass frame
// ---------------------------------------------------------------------
static void translateToCoM(const Vector& masses,
                           const Vector& xs,
                           const Vector& ys,
                           const Vector& zs,
                           Vector& xs_cm,
                           Vector& ys_cm,
                           Vector& zs_cm)
{
    const std::size_t n = masses.size();
    xs_cm.resize(n);
    ys_cm.resize(n);
    zs_cm.resize(n);

    // total molecular mass
    double totalMass = 0.0;
    for (double m : masses) totalMass += m;

    // centre of mass in original coordinates
    double xCM = 0.0, yCM = 0.0, zCM = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        xCM += masses[i] * xs[i];
        yCM += masses[i] * ys[i];
        zCM += masses[i] * zs[i];
    }
    xCM /= totalMass;
    yCM /= totalMass;
    zCM /= totalMass;

    // shift coordinates
    for (std::size_t i = 0; i < n; ++i) {
        xs_cm[i] = xs[i] - xCM;
        ys_cm[i] = ys[i] - yCM;
        zs_cm[i] = zs[i] - zCM;
    }
}

```

This function is used in the code below to construct and diagonalize the moment of inertia matrix.

```cpp
// ---------------------------------------------------------------------
// Build the moment‑of‑inertia tensor (kg·Å²)
// ---------------------------------------------------------------------
static Matrix3 inertiaTensor(const Vector& masses,
                             const Vector& xs,
                             const Vector& ys,
                             const Vector& zs)
{
    // First move to centre‑of‑mass frame
    Vector x_cm, y_cm, z_cm;
    translateToCoM(masses, xs, ys, zs, x_cm, y_cm, z_cm);

    double Ixx = 0.0, Iyy = 0.0, Izz = 0.0;
    double Ixy = 0.0, Iyz = 0.0, Ixz = 0.0;

    const std::size_t n = masses.size();
    for (std::size_t i = 0; i < n; ++i) {
        double m = masses[i];
        double x = x_cm[i];
        double y = y_cm[i];
        double z = z_cm[i];

        Ixx += m * (y*y + z*z);
        Iyy += m * (x*x + z*z);
        Izz += m * (x*x + y*y);
        Ixy += -m * x * y;
        Iyz += -m * y * z;
        Ixz += -m * x * z;
    }

    Matrix3 I;
    I << Ixx, Ixy, Ixz,
         Ixy, Iyy, Iyz,
         Ixz, Iyz, Izz;
    return I;
}

// ---------------------------------------------------------------------
// Principal moments of inertia (kg·m²)
// ---------------------------------------------------------------------
static Vector principalMoments(const Matrix3& I_kg_A2)
{
    // Convert from (kg·Å²) to (kg·m²) by multiplying with (1 Å)² = 1e-20 m²
    const double conversion = atomic_mass_unit / 1e20;   // kg·m² per (Da·Å²)

    // Eigenvalue decomposition (tensor is symmetric)
    Eigen::SelfAdjointEigenSolver<Matrix3> solver(I_kg_A2);
    Vector eigVals(3);
    for (int i = 0; i < 3; ++i) {
        eigVals[i] = solver.eigenvalues()(i) * conversion;
    }

    // Sort in ascending order (Eigen returns them already sorted)
    std::sort(eigVals.begin(), eigVals.end());
    return eigVals;
}

// ---------------------------------------------------------------------
// Rotational constants A, B, C (cm⁻¹)
// ---------------------------------------------------------------------
static std::tuple<double, double, double> rotationalConstants(const std::string& filename)
{
    Vector masses, xs, ys, zs;
    if (!readMoleculeData(filename, masses, xs, ys, zs)) {
        throw std::runtime_error("Failed to read molecular data.");
    }

    Matrix3 I_tensor = inertiaTensor(masses, xs, ys, zs);
    Vector principal = principalMoments(I_tensor);   // kg·m²

    // A = h / (8 π² c I) ; the factor 100 converts m⁻¹ to cm⁻¹
    const double factor = planck_constant / (8.0 * pi * pi * speed_of_light * 100.0);
    double A = factor / principal[0];
    double B = factor / principal[1];
    double C = factor / principal[2];

    return {A, B, C};
}
```

```cpp
// ---------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------
int main()
{
    try {
        const std::string dataFile = "H2O.dat";
        auto [A, B, C] = rotationalConstants(dataFile);
        std::cout << "H2O: A = " << A << " cm-1, "
                  << "B = " << B << " cm-1, "
                  << "C = " << C << " cm-1\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```
```
H2O: A = 27.148870802103488 cm-1, B = 14.654245226373835 cm-1, C = 9.51714245613036 cm-1
```
We have $A\neq B\neq C$, it must be that $I_{A} \neq I_{B} \neq I_{C}$, and $H_{2}O$ is __Asymmetric top__.These values are consistent with published spectroscopy data. [J. Chem. Phys. 24, 1139–1165 (1956)](https://doi.org/10.1063/1.1742731).
