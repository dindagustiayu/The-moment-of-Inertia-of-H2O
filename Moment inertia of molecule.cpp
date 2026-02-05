{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "78a96119-4a9c-41bf-a5c9-4de921f9440d",
   "metadata": {},
   "source": [
    " ---\n",
    "title: \"The Moment Inertia of a Molecule\"\n",
    "date: \"2026-2-4\"\n",
    "categories: [Python 3, Jupyter Notebook, Numpy, Matplotlib, Caycley Table]\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a454897-f4b0-45be-b468-bdcef052f369",
   "metadata": {},
   "source": [
    "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7692151-71b4-426e-be54-d8ac57c28794",
   "metadata": {},
   "source": [
    "# The moment inertia of a molecule\n",
    "\n",
    "## Moment inertia\n",
    "The moment of inertia of a molecule may be defined as the product of mass of each atom and the square of its distance from the rotational axis through the centre of mass of the molecule. Mathematically, it may be written as,\n",
    "<p align='center'>\n",
    "    $$I = \\sum_{i}\\;m_{i}r_{i}^{2}$$\n",
    "</p>\n",
    "where $r_{i}$ is the distance of each atom from the centre of mass.\n",
    "\n",
    "The moment of inertia of a molecule may be resolved into rotational components about three mutually perpendicular directions through the centre of gravity. The quantity $I$, which translates one vector into another vector, is therefore a __matrix__, called the __inertia matrix__:\n",
    "<p align='center'>\n",
    "    $$I=\\begin{pmatrix}I_{xx} & I_{xy} & -I_{xz}\\\\ I_{yx} & I_{yy} & I_{yz}\\\\ I_{zx} & I_{zy} & I_{zz} \\end{pmatrix}$$\n",
    "</p>\n",
    "where the __diagonal__ elements (called \"moment of inertia\") are similar to the one-dimensional definition, e.g.\n",
    "<p align='center'>\n",
    "    $$I_{xx}=\\sum_{i}m_{i}(y_{i}^{2}+z_{i}^{2}),$$\n",
    "    $$I_{yy}=\\sum_{i}m_{i}(x_{i}^{2}+z_{i}^{2}),$$\n",
    "    $$I_{zz}=\\sum_{i}m_{i}(x_{i}^{2}+y_{i}^{2});$$\n",
    "</p>\n",
    "but the __off-diagobal__ elements (called \"products of inertia\") must also be included:\n",
    "<p align='center'>\n",
    "    $$I_{xy}= I_{yx} = \\sum_{i}m_{i}x_{i}y_{i},$$\n",
    "    $$I_{xz}= I_{zx} = \\sum_{i}m_{i}x_{i}z_{i},$$\n",
    "    $$I_{yz}= I_{zy} = \\sum_{i}m_{i}y_{i}z_{i}.$$\n",
    "</p>\n",
    "In the case of a rotation molecule, the index $i$ runs over the atoms in the molecule.\n",
    "Thus, a molecule has three principal moments of inertia, usually designated as $I_{A},\\;I_{B},\\;I_{C}$. The three principal moments of inertia may be taken as,\n",
    "- $I_{A}$ for rotation about the bond axis\n",
    "- $I_{B}$ for end-over-end rotation in the plane of the paper\n",
    "- $I_{C}$ for end-over-end rotation at right angles to the plane of the paper. \n",
    "\n",
    "Based on the values of $I_{A},\\;I_{B}, and\\;I_{C}$, molecules may be classified into several group as:\n",
    "\n",
    "- $I_{A}=0$ while $I_{B}=I_{C}$ : __Linear molecule__, examples: $CO_{2}$, HCl, etc.\n",
    "  \n",
    "- $I_{B}=I_{C}\\neq I_{A}$, while $I_{A}\\neq 0$ : __Symmetric top molecule__.\n",
    "  - (a) if $I_{B}=I_{C} >  I_{A}$ : __Prolate symmetric top molecule__. eg. $CH_{3}Cl$.\n",
    "  - (b) if $I_{B}=I_{C} <  I_{A}$ : __Oblate symmetric top molecule__. eg. $BCl_{3}$.\n",
    "\n",
    "- $I_{A}=I_{B}=I_{C}$ : __Spherical top molecule__. eg. $CH_{4}$\n",
    "\n",
    "- $I_{A}\\neq I_{B}\\neq I_{C}$ : __Asymmetric top molecule__.$CHCl$.\n",
    "\n",
    "Furtheremore, in spectroscopy, it is conventional to define the _rotational constants_, \n",
    "<p align='center'>\n",
    "    $$A=\\frac{h}{8\\pi^{2}cI_{A}},$$\n",
    "    $$A=\\frac{h}{8\\pi^{2}cI_{B}},$$\n",
    "    $$A=\\frac{h}{8\\pi^{2}cI_{C}},$$\n",
    "</p>\n",
    "which are reported in wavenumber units ($cm^{-1}$).\n",
    "\n",
    "The file H2O.dat contains the positions of the atoms in the molecule H2O in XYZ format. Determine the rotational constants for this molecule and classify it as a spherical, oblate, prolate, or asymmetric top.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4065256-381d-461d-9d3f-3d1477a52b95",
   "metadata": {},
   "outputs": [],
   "source": [
    "# H2O\n",
    "# mass\t\tx           \ty           \tz       \n",
    "  15.999\t0.00000000  \t0.00000000  \t0.11779\t\t#O\n",
    "  1.00784\t-0.53418382 \t0.53418382 \t    -0.47116 \t#H\n",
    "  1.00784\t0.53418382 \t    -0.53418382 \t-0.47116 \t#H"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a2439ff-433c-4175-b624-9703044a77be",
   "metadata": {},
   "source": [
    "The four column of the provided data file are mass (in Da) and (x, y, z) coordinates (in $\\mathring{A}$).\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4589423e-245e-4f72-a996-1e48957cdb52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "H2O: A = 27.148870802103488 cm-1, B = 14.654245226373835 cm-1, C = 9.51714245613036 cm-1\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from scipy.constants import u, h, c\n",
    "m, x, y, z = np.genfromtxt('H2O.dat', unpack=True)\n",
    "\n",
    "def translate_to_cofm(m, x, y, z):\n",
    "    \"\"\" Translate the atom positions to be relative to the CofM.\"\"\"\n",
    "\n",
    "    # Total molecular mass.\n",
    "    M = np.sum(m)\n",
    "\n",
    "    # position of center of mass in original coordinates.\n",
    "    xCM = np.sum(m * x) / M\n",
    "    yCM = np.sum(m * y) / M\n",
    "    zCM = np.sum(m * z) / M\n",
    "\n",
    "    # Transform to CoFM coordinates and return them\n",
    "    return x - xCM, y - yCM, z - zCM\n",
    "\n",
    "\n",
    "def get_inertia_matrix(m, x, y, z):\n",
    "    \"\"\" Return the moment of inertia tensor.\"\"\"\n",
    "\n",
    "    x, y, z = translate_to_cofm(m, x, y, z)\n",
    "    Ixx = np.sum(m * (y**2 + z**2))\n",
    "    Iyy = np.sum(m * (x**2 + z**2))\n",
    "    Izz = np.sum(m * (x**2 + y**2))\n",
    "    Ixy = -np.sum(m * x * y)\n",
    "    Iyz = -np.sum(m * y * z)\n",
    "    Ixz = -np.sum(m * x * z)\n",
    "    I = np.array([[Ixx, Ixy, Ixz],\n",
    "                  [Ixy, Iyy, Iyz],\n",
    "                  [Ixz, Iyz, Izz]\n",
    "                 ])\n",
    "    return I\n",
    "\n",
    "def get_principal_moi(I):\n",
    "    \"\"\" Determine the principal moments of inertia.\"\"\"\n",
    "\n",
    "    # The principal moments of inertia are the eigenvalues of the moment of inertia tensor.\n",
    "    Ip = np.linalg.eigvals(I)\n",
    "\n",
    "    # Sort and convert principal moments of inertia to kg.m2 before returning.\n",
    "    Ip.sort()\n",
    "    return Ip * u / 1e20\n",
    "\n",
    "def get_rotational_constants(filename):\n",
    "    \"\"\" Return the rotational constants, A, B, C (in cm-1) for a molecule.\n",
    "\n",
    "    The atomic coordinates are retrieved from filename which should have\n",
    "    four columns of data: mass (in Da), and x, y, z coordinates (in Angstrong).\n",
    "\n",
    "    \"\"\"\n",
    "    m, x, y, z = np.genfromtxt(filename, unpack=True)\n",
    "    I = get_inertia_matrix(m, x, y, z)\n",
    "    Ip = get_principal_moi(I)\n",
    "    A, B, C = h / 8 / np.pi**2 / c / 100 / Ip\n",
    "    return A, B, C\n",
    "\n",
    "A, B, C = get_rotational_constants('H2O.dat')\n",
    "print(f'H2O: A = {A} cm-1, B = {B} cm-1, C = {C} cm-1')\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a03ea6a7-b82c-4041-ad20-6dd258e9037d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eca0094d-3ba3-4d72-b4a9-ff121e56c5df",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
