import os
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
import scipy.constants as sc

################################################################
# Unless explicitly stated otherwise, all units follow the SI. #
################################################################

# Turn interactive mode on for pyplot.
plt.ion()

# Initialize an array to store altitude values.
altitude = np.linspace(0, 100000, 201)


###
# Calculate the acceleration due to gravity 'g' as a function of altitude
###


# According to the International Astronomical Union, the mass of the Earth times
# the gravitational constant, called the nominal terrestrial mass parameter:
GM_e = 3.986004e14
# Again, according to the IAU, the equatorial radius of the Earth:
R_e = 6.3781e6

# Exact calculation of g according to Newton's universal law of gravitation.
g_exact = np.divide(GM_e, np.square(R_e + altitude))

# Calculation via first order Taylor expansion:
g_approx = np.multiply((GM_e / (R_e**2)), 1 - np.multiply(2, np.divide(altitude, R_e)))

print(' ')
# Ask whether to show the graphs for both values of g or not.
show_g_graphs = input('Show graphs of g vs altitude? (y/n): ')

if show_g_graphs == str('y'):
    plt.figure(0, frameon=False)
    plt.title('Acceleration due to gravity as a function of altitude')
    plt.plot(altitude, g_exact, 'ro', markersize=1.3, label='Exact calculation.')
    plt.plot(altitude, g_approx, 'go', markersize=1.3, label='First order Taylor expansion')
    plt.xlabel('Altitude ' + r'$[m]$')
    plt.ylabel('g ' + r'$\left[\dfrac{m}{s^2}\right]$')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
elif show_g_graphs == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

'''
Respuesta punto 1.

Dependiendo de la precision que se busque. Si se quiere un calculo mas bien de orden de magnitud, es una buena aproximacion.
Pero si se busca un calculo preciso, creo que no es buena.
'''

###
# Calculate the mass of the atmosphere.
###

# Store standard atmospheric pressure.
std_atm_pressure = sc.physical_constants['standard atmosphere'][0]

# Calculate total mass of the atmosphere.
m_atmosphere = (std_atm_pressure / 9.8) * 4 * sc.pi * (6370000**2)

print(' ')
# Ask whether to print the mass of the atmosphere or not.
show_atm_mass = input('Print mass of the atmosphere? (y/n): ')

if show_atm_mass == str('y'):
    print('The total mass of the atmosphere is: ' + str(m_atmosphere) + 'Kg.')
elif show_atm_mass == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Pressure as a function of altitude.
###

# First, assuming a constant temperature distribution of 273K.
gas_constant = sc.physical_constants['molar gas constant'][0]
scale_height = (gas_constant * 273) / (9.8 * 0.02896)
pressure_ctd = std_atm_pressure * np.exp(-np.divide(altitude, scale_height))

# Now, using a piecewise linear temperature distribution.
# Initialize an array to store temperature as a function of altitude.
temp_distribution = np.zeros(201)

# Define the pieces of the piecewise distribution.
# Initialize an 'x' array. It's initialized by the arange function because
# of the way the select function works.
x = np.arange(201)
# These are the intervals over which the distribution is defined.
cond_list = [(0 <= x) & (x < 20), (20 <= x) & (x < 30), (30 <= x) & (x < 100),
            (100 <= x) & (x < 170), (170 <= x) & (x <= 200)]
# These are the linear functions that make up the temperature distribution.
# Here, x/2 is altitude in Km, so slopes are in K/km.
choice_list_t = [-5*(x/2) + 273,
                223,
                (10/7)*(x/2) + (1411/7),
                (-20/7)*(x/2) + (2911/7),
                4*(x/2) - 167]

# Fill the temperature distribution array.
temp_distribution = np.select(cond_list, choice_list_t)

print(' ')
# Ask whether to show the temperature distribution or not.
show_t_dist = input('Plot temperature distribution? (y/n): ')

if show_t_dist == str('y'):
    plt.figure(1, frameon=False)
    plt.title('Temperature distribution as a function of altitude')
    plt.plot(altitude, temp_distribution, 'go', markersize=1.3)
    plt.xlabel('Altitude ' + r'$[m]$')
    plt.ylabel('Temperature ' + r'$[K]$')
    plt.yticks(list(plt.yticks()[0]) + [273])
    plt.grid(True)
    plt.show()
elif show_t_dist == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

# Initialize array to store pressure values.
pressure_ltd = np.zeros(201)

# Define things necessary to fill the pressure array.
# First store slope and constant term for all linear functions of the
# temperature distribution. Here, slopes are in K/m.
s_and_c = [-5/1000, 273, 0, 223, 1/700, 1411/7, -1/350, 2911/7, 1/250, -167]

# Store initial pressure and initial temperature over the intervals defined
# above for the temperature distribution (cond_list).
p_and_t = [std_atm_pressure, 273, 0, 223, 0, 223, 0, 273, 0, 173]

for i in range(1, 5):
    try:
        p_and_t[2*i] = p_and_t[2*i-2]*np.exp(((-9.8 * 0.02896)/(gas_constant * s_and_c[2*i-2]))*np.log(p_and_t[2*i+1]/p_and_t[2*i-1]))
    except ZeroDivisionError:
        p_and_t[2*i] = p_and_t[2*i-2]*np.exp(((-9.8 * 0.02896)/(gas_constant * s_and_c[2*i-1]))*5000)

# Function: pressure
# Calculates the pressure at a certain altitude.
# Parameters:
#   index - this is the index of the element of the pressure array being
#           calculated.
#   slope - this is the slope of the linear function that represents the
#           temperature distribution over the interval that corresponds to
#           said index.
#   constant - this is the linear term of the linear function of the slope.
#   p_i - initial pressure of the interval (i.e. at the lower endpoint).
#   t_i - initial temperature of the interval.
#   temp (Default) - the temperature array.
def pressure(index, slope, constant, p_i, t_i, temp = temp_distribution):
    g = 9.8
    M = 0.02896
    R = sc.physical_constants['molar gas constant'][0]
    if slope == 0:
        exponent_b = (g * M) / (R * constant)
        # Here index*500 is the altitude in meters.
        return p_i * np.exp(-exponent_b*(index*500 - 10000))
    exponent_a = (g * M) / (R * slope)
    return p_i * np.exp(-exponent_a * np.log(np.abs( temp[index] / t_i) ))

# These are the expressions of the pressure over the intervals defined
# above for the temperature distribution (cond_list).
choice_list_p = [pressure(x, s_and_c[0], s_and_c[1], p_and_t[0], p_and_t[1]),
                 pressure(x, s_and_c[2], s_and_c[3], p_and_t[2], p_and_t[3]),
                 pressure(x, s_and_c[4], s_and_c[5], p_and_t[4], p_and_t[5]),
                 pressure(x, s_and_c[6], s_and_c[7], p_and_t[6], p_and_t[7]),
                 pressure(x, s_and_c[8], s_and_c[9], p_and_t[8], p_and_t[9])]

# Fill the pressure array.
pressure_ltd = np.select(cond_list, choice_list_p)

print(' ')
# Ask whether to show the pressure graphs or not.
show_pressure = input('Plot pressure vs altitude? (y/n): ')

if show_pressure == str('y'):
    # Graph with logarithmic on the pressure axis.
    plt.figure(2, frameon=False)
    plt.title('Pressure as a function of altitude.')
    plt.plot(altitude, pressure_ctd, 'bo', markersize=1.3, label = 'Constant temperature distribution.')
    plt.plot(altitude, pressure_ltd, 'ro', markersize=1.3, label = 'Piecewise linear temperature distribution.')
    plt.xlabel('Altitude ' + r'$[m]$')
    plt.ylabel('Pressure ' + r'$[Pa]$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    # Graph with logarithmic scale on both axes.
    plt.figure(3, frameon=False)
    plt.title('Pressure as a function of altitude.')
    plt.plot(altitude, pressure_ctd, 'bo', markersize=1.3, label = 'Constant temperature distribution.')
    plt.plot(altitude, pressure_ltd, 'ro', markersize=1.3, label = 'Piecewise linear temperature distribution.')
    plt.xlabel('Altitude ' + r'$[m]$')
    plt.ylabel('Pressure ' + r'$[Pa]$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()
elif show_pressure == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Calculate the percentage of atmosphere mass in the troposphere and
# stratosphere using both temperature distributions defined above.
###

# Function: mass_percentage
# Returns the percentage of atmospheric mass bound between two altitude
# values.
# Parameters:
#   p_i - pressure at the lower endpoint.
#   p_f - pressure at the higher endpoint.
def mass_percentage(p_i, p_f):
    # Store the standard atmospheric pressure.
    p_0 = sc.physical_constants['standard atmosphere'][0]

    return (p_i-p_f)/p_0

# For the constant temperature distribution.
mp_troposphere_ctd = mass_percentage(pressure_ctd[0], pressure_ctd[24])
mp_stratosphere_ctd = mass_percentage(pressure_ctd[24], pressure_ctd[100])

# For the piecewise linear temperature distribution.
mp_troposphere_ltd = mass_percentage(pressure_ltd[0], pressure_ltd[24])
mp_stratosphere_ltd = mass_percentage(pressure_ltd[24], pressure_ltd[100])

# Calculate percentage difference between both values.
pd_troposphere = np.abs(mp_troposphere_ctd - mp_troposphere_ltd) / np.sqrt(mp_troposphere_ctd + mp_troposphere_ltd)
pd_stratosphere = np.abs(mp_stratosphere_ctd - mp_stratosphere_ltd) / np.sqrt(mp_stratosphere_ctd + mp_stratosphere_ltd)

print(' ')
# Ask whether to show the mass percentage calculations or not.
show_m_percentage = input('Show mass percentage calculations? (y/n): ')

if show_m_percentage == str('y'):
    print(' ')
    print('For a constant temperature distribution of 273K:')
    print('Percentage mass bounded by the troposphere: ' + str(mp_troposphere_ctd))
    print('Percentage mass bounded by the stratosphere: ' + str(mp_stratosphere_ctd))
    print(' ')
    print('For the piecewise linear temperature distribution shown before:')
    print('Percentage mass bounded by the troposphere: ' + str(mp_troposphere_ltd))
    print('Percentage mass bounded by the stratosphere: ' + str(mp_stratosphere_ltd))
    print(' ')
    print('The percentage difference between these two calculations are:')
    print('Percentage difference for the troposphere: ' + str(pd_troposphere))
    print('Percentage difference for the stratosphere: ' + str(pd_stratosphere))
elif show_m_percentage == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Calculate molecule concentration as a function of pressure and temperature.
# Calculate molecules per unit area between two altitudes.
###


# Function: molecule_concentration
# Returns molecule concentration (number of particles/volume) in the air.
# Parameters:
#   pressure - pressure value.
#   temperature - temperature value.
def molecule_concentration(pressure, temperature):
    N_A = sc.physical_constants['Avogadro constant'][0]
    R = sc.physical_constants['molar gas constant'][0]

    return (N_A * pressure) / (R * temperature)

# Molecule concentration at sea level.
mc_std = molecule_concentration(pressure_ltd[0], temp_distribution[0])

# Molecule concentration at 10,000 meters.
mc_10km = molecule_concentration(pressure_ltd[20], temp_distribution[20])

# Calculate molecules of air per unit area between 20km and 25km.
N_A = sc.physical_constants['Avogadro constant'][0]
M_air = 0.02896
g = 9.8
mpa_air = (N_A / (M_air * g)) * (pressure_ltd[40] - pressure_ltd[50])

# Calculate molecules of oxygen per unit area between 20km and 25km.
M_o2 = 0.03198 # Molar mass of diatomic oxygen.
X_o2 = 0.2095  # Molar fraction of diatomic oxygen in the atmosphere
w_o2 = X_o2 * (M_o2 / M_air) # Mass fraction of diatomic oxygen in air.
mpa_o2 = ((w_o2 * N_A) / (M_o2 * g)) * (pressure_ltd[40] - pressure_ltd[50])

print(' ')
# Ask whether to show the concentration calculations or not.
show_concentration = input('Show concentration calculations? (y/n): ')

if show_concentration == str('y'):
    print(' ')
    print('At sea level there are ' + str(mc_std) + ' molecules per unit volume.')
    print(' ')
    print('At 10,000 meters there are ' + str(mc_10km) + ' molecules per unit volume.')
    print(' ')
    print('Bounded between 20,000 meters and 25,000 meters there are:')
    print(str(mpa_air) + ' air molecules per unit area.')
    print(str(mpa_o2) + ' oxygen molecules per unit area.')
elif show_concentration == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')


###
# Water vapor calculations.
###


# Function: sat_pressure
# Returns the water vapor saturation pressure for a given temperature.
# Parameters:
#   temperature - temperature value.
def sat_pressure(temperature):
    R = sc.physical_constants['molar gas constant'][0]
    L_v = 2264705 # Specific latent heat of vaporization of water in J/kg
    M_v = 0.0180153 # Molar mass of water.
    P_s0 = 120 * np.exp((L_v * M_v) / (R * 253))

    return P_s0 * np.exp(-(L_v * M_v) / (R * temperature))

# Function: vap_density_e
# Returns water vapor density as a function of relative humidity and
# temperature. Uses an empirical fit to calculate the saturation vapor
# density of water. This fit gives accurate results only between 273K
# and 313K.
# Parameters:
#   rel_h - relative humidity.
#   temp - temperature.
def vap_density_e(rel_h, temp):
    # Calculate saturation vapor density using fit.
    coef = [5.018, 0.32321, 8.1847e-3, 3.1243e-4]
    # Fit works for temperature in degree Celsius. Output in g/m^3.
    svd = np.polynomial.polynomial.polyval(temp - 273, coef)

    return rel_h * svd / 1000

# Calculate water vapor density for Antarctica; 100% relative humidity and 273K.
wvp_antartica_e = vap_density_e(1, 273)

#Calculate water vapor density for tropical desert; 10% humidity and 313K
wvp_desert_e = vap_density_e(0.1, 313)

# Function: vap_density_t
# Returns water vapor density as a function of relative humidity and
# temperature. This is via the ideal gas equation and using the
# definition of relative humidity (i.e. pressure/saturation pressure).
# Parameters:
#   rel_h - relative humidity.
#   temp - temperature.
def vap_density_t(rel_h, temp):
    R = sc.physical_constants['molar gas constant'][0]
    M_v = 0.0180153 # Molar mass of water.
    return ((rel_h * M_v)/(R * temp)) * sat_pressure(temp)

# Calculate water vapor density for Antarctica; 100% relative humidity and 273K.
wvp_antartica_t = vap_density_t(1, 273)

#Calculate water vapor density for tropical desert; 10% humidity and 313K
wvp_desert_t = vap_density_t(0.1, 313)

print(' ')
# Ask whether to show the water vapor calculations or not.
show_vapor = input('Show water vapor densities? (y/n): ')

if show_vapor == str('y'):
    print(' ')
    print('Results using an empirical fit for saturation vapor density of water:')
    print('For Antarctica there are ' + str(wvp_antartica_e) + ' kilograms of water vapor per cubic meter.')
    print('For the tropical desert there are ' + format(wvp_desert_e, '.7f') + ' kilograms of water vapor per cubic meter.')
    print(' ')
    print('Results via the ideal gas equation:')
    print('For Antarctica there are ' + format(wvp_antartica_t, '.7f') + ' kilograms of water vapor per cubic meter.')
    print('For the tropical desert there are ' + format(wvp_desert_t, '.7f') + ' kilograms of water vapor per cubic meter.')
elif show_vapor == str('n'):
    pass
else:
    sys.exit('Script stopped. Must input "y" or "n".')

print(' ')
end = input('Press enter to end the script.')
