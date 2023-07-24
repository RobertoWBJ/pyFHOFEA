# pyFHOFEA

⚠️ **WARNING - The code is under development, so some inconsistency may appear.**

# INTRODUCTION

The pyFHOFEA is a python package developed to analyse the stresses and strains acting on any plate geometry by the finite element method.

It was developed to obtain the graduation degree in the area of Mechanical Engineering.

The code uses the same logic of any finite element software to develop an analysis:
* Pre-processing (define geometry and boundary conditions);
* Processing (get the results by solving the linear system);
* Post-processing (show graphically the requested results by the user).

The script only handles the *Constant Strain Triangle* element, so the mesh must be set as *all triangles*. It solves by considering the elastic regime of ductile materials and small deformations.

## Packages used

The script uses the following packages:
* [meshio](https://github.com/nschloe/meshio) (version 5.3.4);
* [matplotlib](https://matplotlib.org/3.5.1/index.html) (version 3.5.1);
* [numpy](https://numpy.org/doc/) (version 1.22.4).


# Pre-processing

## Creating the plane geometry

The script uses the Gmsh software to design the geometry and set the *physical groups* that will be used to define the boundary conditions. The Gmsh can be obtained clicking [here](https://gmsh.info/).

## Importing the plane geometry

First, initialize the *Pre_processing* class by informing the directory of the *.msh* geometry.

## Setting the boundary conditions

To remember the names of the *physical groups* in the geometry use the attribute:
*```_.physical_groups```.

To apply the forces, use the method ```apply_forces()```. The first parameter is the name of the *physical group* where the force will be applied, the second parameter is the force in the X direction and the last parameter is the force in the Y direction.

**IMPORTANT:** the forces can be applied on nodes or edges only.

To set the restrictions, use the method ```set_restrictions()```. The first parameter is the name of the *physical group* where the constraint will be applied, the second parameter is the translation DOF in the X direction and the last parameter is the translation DOF in the Y direction. Type 0 to constraint the DOF or 1 to maintain the DOF.
 
To define the thickness of the plate and the material properties (young modulus and poisson coefficient) use the method ```set_physical_properties()```.

**IMPORTANT:** the parameters informed on the method ```set_physical_properties()``` are used in all the extension of the imported geometry.

# Processing

Initialize the *Processing* class by informing the *Pre_processing* instance.

To find the displacement by the equation ${F} = [K]{d}$, the user must solve the linear system using the ```solve()``` method.

The global stiffness matrix is assembled in the beginning of the function used to solve the linear system and the results (stresses and strains) are calculated by the method ```get_all_results()``` (this command must be used after the ```solve()```).

# Post processing

To initialize this class, the user must the the set the instances used in the *Pre_processing* and *Processing* classes.

The results available are:
* Normal strain in X direction;
* Normal strain in Y direction;
* Shear strain in XY plane;
* Stress in X direction;
* Stress in Y direction;
* Stress in XY plane;
* Maximum principal;
* Minimum principal;
* Maximum shear stress on plane;
* von Mises stress.

To check the results, the user should call the respective function and set the scale if necessary.
If the user has named a region by a *physical group*, just this area can be displayed by informing its name as a parameter for the plot method.

# Last comments

A notebook showing a complete practical example of an analysis can be checked on the files available on the top of this page.

Feel free to make suggestions, share bugs and contact me.
