'''
List of functions copied from the previous workbooks for our semi-analytic model.
'''

### Imports
import os
import pickle
import numpy as np
from scipy import interpolate as interp
from scipy.optimize import brentq
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

### File paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TREE_FILE = os.path.join(DATA, 'tree.npy')
SCALES_FILE = os.path.join(HERE, 'scales.p')
COOLING_FILE = os.path.join(DATA, 'Cooling', 'RadLoss_-0.50_0.00_0.00_1.00e+00.rates')


### Constants
h = 0.73
Om = 0.25
G = 4.3022682e-9
H = 100
mu = 0.59
mp = 1
fH = 0.75
kB = 8.31445986e-3
amu_kms2_to_erg = 1.66053904e-14
Mpc_over_kms_to_yr = 9.77813106e11
s_per_yr = 3.1556925e7
Mpc_per_cm = 3.2407793e-25
Msun_per_amu = 8.3481929e-58
fb = 0.18


### Workbook 2
def rho_vir(z): #rho_vir in (Msun/h) / (Mpc/h)^3
    Om_z = Om*((1+z)**3)/(Om*(1+z)**3+(1-Om)) #Omega_Matter(z)
    x = Om_z-1
    delta_c = 18*(jnp.pi**2)+82*x-39*x*x #virial overdensity wrt critical density, from Bryan & Norman (1998)
    rho_matter_z0 = Om*3*((100)**2)/(8*jnp.pi*G)  #rho_m(z=0) in (Msun/h) / (Mpc/h)^3
    rho_matter_z = rho_matter_z0*((1+z)**3)      #rho_m(z) in (Msun/h) / (Mpc/h)^3
    rho_crit = rho_matter_z / Om_z               #in (Msun/h) / (Mpc/h)^3
    return delta_c*rho_crit                      #in (Msun/h) / (Mpc/h)^3

def t_dyn(z): #Dynamical time, in units of (Mpc/h) / (km/s)
    return 1/jnp.sqrt(4*jnp.pi*G*rho_vir(z)/3)

def R_vir(z,M): #M in units of Msun/h, returns radius in Mpc/h
    return ((M/(4*jnp.pi/3*rho_vir(z)))**(1./3.))

def T_vir(z,M): #Halo virial temperature, in K
    return mu*mp/(2.0*kB)*G*M/R_vir(z,M)

def sim_units_to_years(t): #t in units of (Mpc/h) / (km/s); returns units of years
    return (Mpc_over_kms_to_yr*t/h)


cooling_func_with_metals = np.loadtxt(COOLING_FILE, skiprows=2)
cooling_func_spline = interp.UnivariateSpline(cooling_func_with_metals[:,0], cooling_func_with_metals[:,1], s=0)


### Define a function for r cool, assuming the above-defined cosntants
def r_cool(m_hot, z, M):
    ### Need to convert its units as well!!!
    ### otherwise it would return r_cool in (erg/s cm^3)
    m_hot_amu = m_hot / (h * Msun_per_amu)
    t_dyn_s = t_dyn(z) * Mpc_over_kms_to_yr * s_per_yr / h
    R_vir_cm = R_vir(z, M) / (h * Mpc_per_cm)
    T = T_vir(z,M)
    kT_erg = kB * T * amu_kms2_to_erg
    ### finally we can use the above equation
    return np.sqrt((fH * m_hot_amu * t_dyn_s * cooling_func_spline(T)) \
                   / (6 * np.pi * R_vir_cm * mu * mp * kT_erg))

#What you have to write:
def mass_cooling_rate(z,M,m_hot):
    #Remember that everything has to be converted to the correct units(!)
    #Helpful to sanity check against the graph of t_cool / t_Universe in the class notebook.
    ### r_cool is in cm, so R_vir has to be in cm too for the ratio to be dimensionless
    R_vir_cm = R_vir(z, M) / (h * Mpc_per_cm)
    return (m_hot * r_cool(m_hot, z, M)) / sim_units_to_years((t_dyn(z)) * R_vir_cm)


def mass_cooling_rate_spline(z):
    halo_mass = 10**np.arange(8,16,0.05) #From 10^8 Msun/h to 10^16 Msun/h
    cooling_rates = np.array([mass_cooling_rate(z,m,1) for m in halo_mass])
    return interp.UnivariateSpline(halo_mass, cooling_rates, s=0) #You fill in the ... part!


def fast_mass_cooling_rate(spline,M,m_hot):
    return ((m_hot**(3/2))*spline(M)) #How easy is that!


def exact_time_to_scale(time_in_years):
    t = time_in_years*(h*100/Mpc_over_kms_to_yr) #t needs to be in units of 1/H
    m = (np.sinh(1.5*t*np.sqrt(1-Om)))**2;
    return (((Om*m)/(1-Om))**(1/3))


def scale_to_time_spline():
    t = np.arange(0,14e9,1e7)
    scales = [exact_time_to_scale(x) for x in t]
    return interp.UnivariateSpline(scales,t)


### Workbook 3
def re_stars(r_halo): #from https://arxiv.org/pdf/1212.2980.pdf
    return 0.015 * r_halo #in comoving kpc/h


def re_gas(r_halo): #from https://arxiv.org/pdf/1212.2980.pdf
    return 2.7 * re_stars(r_halo) #in comoving kpc/h


def rs(re):
    ### M_ratio = 1 - (1 + x) * np.exp(-x)
    ### x_half at M_ratio=0.5:
    ### 0.5 = 1 - (1 + x) * np.exp(-x)
    ### cant be solved by hand, so rearrange and use scipy:
    f = lambda x: (1 + x) * np.exp(-x) - 0.5
    x_half = brentq(f, 0.1, 10)
    ### this value is ~1.68, I checked it outside this function
    return re / x_half #in comoving kpc/h


def Sigma_0(mass,rs):
    return mass / (2 * np.pi * rs**2) #in Msun/h / (comoving kpc/h)^2


def disk_sfr(Sigma_0, rs, scale_factor):
    #Don't forget that the Kennicutt-Schmidt law is in *physical* units, whereas the simulation units are *comoving*
    #If "x" is a comoving distance, then the physical distance is scale_factor*x
    ### from section 5: Sigma*dot = 2.5e-4 Sigma_{gas}^1.4 (no error bars)
    ### so we get the integral of A * e^(-1.4 r/rs)r dr
    ### plugging into an integral calculator:
    ### A * 25 rs^2 / 49 and A = 2pi * 2.5e-4 * Sigma_0^1.4
    ### converting lengths to comoving scales with using the scale factor
    rs_phys = scale_factor * rs / h
    ### now for Sigma, noting that it goes 1/r^2
    Sigma_0_phys = (Sigma_0 * h/ scale_factor**2) / 1e6

    return  2 * np.pi * 2.5e-4 * Sigma_0_phys**1.4 * rs_phys**2 * 25/49 * h #in Msun/h / yr



### Workbook 4
def KS_Constant():
    re = 1 #Gas effective radius, in physical kpc/h
    gas_rs = rs(re) #Gas scale radius, in physical kpc/h
    gas_mass = 1 #Msun/h
    gas_sigma_0 = Sigma_0(gas_mass, gas_rs) #Gas surface density, in Msun/h / (physical kpc/h)^2
    return 1.0/disk_sfr(gas_sigma_0, gas_rs, 1) #Call the K-S law at a=1 so that physical and comoving 
                                                # units are the same


def KS_Cold_Gas(sfr,re_gas,scale_factor,ks_constant):
    ### need to convert comoving to physical units
    re_gas_phys = re_gas * scale_factor
    m_cold = (ks_constant * sfr * re_gas_phys**0.8)**(1/1.4) #You fill this out!
    return m_cold


def supernovae_reheating_ratio(t_vir):
    energy_available = sn_reheat_fraction*sn_energy #Units of ergs / solar mass of stars
    ### From thermodynamics, the equation is (3/2 Kb T) / (mu mp)
    gas_thermal_energy = t_vir * 1.5*kB/(mu*mp) * amu_kms2_to_erg/Msun_per_amu
    #Should be in units of ergs / solar mass of gas
    return energy_available / gas_thermal_energy


def supernovae_ejection_ratio(v_max):
    energy_available = sn_eject_fraction*sn_energy #Units of ergs / solar mass of stars
    kms2_to_erg_per_amu = amu_kms2_to_erg #note that both sides are divided by the same unit
    #float64 avoids a float32 overflow: NumPy>=2.0 keeps float32_field*python_float as float32
    ejection_energy = (np.float64(v_max)**2)*kms2_to_erg_per_amu/Msun_per_amu #ergs / Msun, using virial theorem approximation
    return energy_available / ejection_energy


def calc_t_vir_spline():
    z = np.linspace(0,20,200) #Redshift, from 0-20
    t_vir = T_vir(z,1) #Virial temperature of a 1 Msun/h halo for z=0..20
    return interp.UnivariateSpline(z,t_vir,s=0)

t_vir_spline = calc_t_vir_spline()


def fast_T_vir(z,M):
    return t_vir_spline(z)*(M**(2./3.))


sn_energy = 1e51*5e-3 # ergs/Msun: energy in supernovae per unit stellar mass formed
                      #   from Eq. 7 in https://arxiv.org/abs/1004.2518

### Im putting my max sn feedback values, which were 50/50 for reheat/ejection
sn_reheat_fraction = 0.5 #fraction of that energy that heats the cold gas
sn_eject_fraction = 0.5    #fraction of that energy that expels gas from the halo


### Workbook 5
radiative_efficiency = 0.1
kinetic_coupling_to_cold_gas = 0.1


def eddington_rate(m_bh):  #m_bh in Msun/h
    ### using the value as before with epsilon = 0.1
    salpeter_time = 5e7 #in years, depends on the radiative efficiency
    rate = m_bh / salpeter_time #You fill this in, in Msun/h/yr
    return rate


def bh_kinetic_power(m_bh,m_bh_acc_rate):
    edd_rate = eddington_rate(m_bh) #Msun/h/yr
    nonzero_edd_rate = np.where(edd_rate>0, edd_rate, 1.) #Avoids dividing by zero for halos without a black hole
    edd_ratio = m_bh_acc_rate / nonzero_edd_rate #dimensionless
    speed_of_light_squared = 5.66504458e46 #erg/s/(Msun/yr)
    # power = ... # in units of your choice -- make sure to document!
    # high_power = ... #alternate power prescription for high accretion rates
    ### erg/s
    power = radiative_efficiency/(1 - radiative_efficiency) * speed_of_light_squared * m_bh_acc_rate 
    high_power = 0.01*radiative_efficiency/(1 - radiative_efficiency) * speed_of_light_squared * edd_rate  #erg/s

    #np.where(condition, a, b) acts like "if condition: a, else: b" for every element of an array at once
    power = np.where(edd_ratio > 0.01, high_power, power)
    return np.where(m_bh>0, power, 0.) #No black hole, no power


def bh_reheating_rate(m_bh,m_bh_acc_rate,temp): #m_bh in Msun/h, m_bh_acc_rate in Msun/h/yr, temp in K
    power = kinetic_coupling_to_cold_gas*bh_kinetic_power(m_bh, m_bh_acc_rate)*s_per_yr #ergs/yr
    ### stat mech: U = 3/2 kB T
    ### this was taken from Workbook 4
    gas_thermal_energy = temp * 1.5*kB/(mu*mp) * amu_kms2_to_erg/Msun_per_amu
    return power / gas_thermal_energy #Msun/h/yr


def bh_fast_mass_reheating_rate(m_bh, temp):
    edd_rate = eddington_rate(m_bh)
    return bh_reheating_rate(m_bh, 0.01*edd_rate, temp)


def bh_slow_mass_loading_factor(temp):
    slow_accretion_rate = 0.01*eddington_rate(1) #Msun/h/yr for 1 Msun/h black hole
    slow_reheating_rate = bh_reheating_rate(1,slow_accretion_rate,temp) #Low-luminosity reheating rate, Msun/h/yr
    slow_cooling_rate = slow_accretion_rate / alpha #Cooling rate needed to support BH accretion rate, Msun/h/yr
    f_BH = slow_reheating_rate / slow_cooling_rate  #Mass loading factor
    return f_BH


f_ICL = 0.8       #fraction of merging satellite stars that goes into the intracluster light
alpha = 0.001     #Ratio between BH mass and galaxy mass
r_bulge = 0.05    #Ratio of incoming merger mass that will trigger bulge formation
inverse_G = 232435.533 #1/G in (Msun/h) / (physical kpc/h) / (km/s)^2; used for the maximum disk mass
                       #source: Google search for (km/s)^2 / (G*solar mass/kpc)


### Workbook 6
### for the disk instability section
disk_instability = False

halos_at_scale = pickle.load(open(SCALES_FILE, 'rb')) #Indices of the halos at each scale factor
scales = sorted(halos_at_scale.keys())
s_to_time = scale_to_time_spline()
ks_constant = KS_Constant()


