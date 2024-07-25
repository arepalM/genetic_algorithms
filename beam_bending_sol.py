import random

L   = 10000   # Length of beam (mm)
P   = 100     # Concentrated force (N)
La  = 5000    # Location of concentrated force from left support (mm)
Lb  = L-La    # Location of concentrated force from right support (mm)
x   = La      # Location of interest for maximum deflection (mm)
rho = 7.85E-6 # Density (kg/mm^3)

PPKG = 10     # $/kg cost of material

# Simply supported beam optimizable parameters #
E_range = [10E3, 200E3]    # Young's Modulus (MPa)
b_range = [10, 250]        # web thickness 
B_range = [5, 500]         # flange width
h_range = [5, 250]         # flange thickness
H_range = [10, 500]        # web height

Emin = E_range[0] 
Emax = E_range[1]
bmin = b_range[0]
bmax = b_range[1]
Bmin = B_range[0]
Bmax = B_range[1] 
hmin = h_range[0]
hmax = h_range[1]
Hmin = H_range[0]
Hmax = H_range[1]

E = random.uniform(Emin, Emax)
Wt = random.uniform(bmin, bmax)
TFw = random.uniform(Bmin, Bmax)
TFt = random.uniform(hmin, hmax)
Wh = random.uniform(Hmin, Hmax)

BFt = TFt
BFw = TFw
Ixx = (TFw*(TFt+TFt+Wh)**3-(TFw-Wt)*(TFt+TFt+Wh-2*TFt)**3)/12

d1 = (P*Lb*x)/(6*L*E*Ixx) * (L**2 - Lb**2 - x**2)
print(d1)



