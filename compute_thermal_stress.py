"""
Compute the thermal elastic stress as functions of depth using the equations in Berger (1975)
"""

###
# Import the necessary libraries
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import abs, deg2rad, rad2deg, sqrt, sum, nan, logspace, pi, exp, log10, real
from matplotlib.pyplot import subplots
from matplotlib import colormaps
from matplotlib.patches import Rectangle

from utils_basic import PHYS_DIR as dirpath_phys, NUM_SEONCDS_IN_DAY as num_seconds_in_day, BAR as bar
from utils_plot import save_figure

###
# Define the functios
###

# Get the Gamma factor 
def get_gamma(omega, k, kappa, simplify=False):
    """
    Compute the Gamma factor for the thermal elastic stress
    """
    if simplify:
        gamma = (1 + 1j) * sqrt(omega / 2 / kappa)
    else:
        gamma = k * sqrt( 1 + 1j * omega / kappa / k ** 2)
        
    return gamma

# Get the strain 
def get_strain(gamma, k, depths, beta, sigma, temp, simplify=False):
    """
    Compute the strain for the thermal elastic stress as a function of depth
    """
    if simplify:
        epsilons_xx = (1 + sigma) / (1 - sigma) * k / gamma * ((2 * (1 - sigma) - k * depths) * exp(-k * depths) - k / gamma * exp(-gamma * depths)) * beta * temp
        epsilons_yy = (1 + sigma) / (1 - sigma) * ( - k / gamma * (2 * sigma - k * depths) * exp(-k * depths) + exp(-gamma * depths)) * beta * temp
    else:
        epsilons_xx = (1 + sigma) / (1 - sigma) * k / (gamma + k) * (( 2 * (1 - sigma) + k / (gamma - k) - k * depths ) * exp(-k * depths) - k / (gamma - k) * exp(-gamma * depths)) * beta * temp
        epsilons_yy = (1 + sigma) / (1 - sigma) * k / (gamma + k) * ((-2 * sigma - k / (gamma - k) + k * depths) * exp(-k * depths) + gamma ** 2 / k / (gamma - k) * exp(-gamma * depths)) * beta * temp

    return epsilons_xx, epsilons_yy


###
# Input arguments
###
parser = ArgumentParser()
parser.add_argument('--lambs', type=float, nargs='+', help='The surface-temperature horizontal wavelengths to consider')

parser.add_argument('--kappa', type=float, default=1e-6, help='The thermal diffusivity')
parser.add_argument('--beta', type=float, default=1e-5, help='The thermal expansion coefficient')
parser.add_argument('--sigma', type=float, default=0.25, help='Poisson ratio')
parser.add_argument('--vp', type=float, default=4000.0, help='The P-wave velocity in m/s')
parser.add_argument('--rho', type=float, default=2500.0, help='The density in kg/m^3')
parser.add_argument('--temp', type=float, default=10.0, help='The amplitude of surface temperature variation')
parser.add_argument('--max_depth', type=float, default=1000.0, help='The maximum depth to consider')
parser.add_argument('--num_depths', type=int, default=100, help='The number of depths to consider')
parser.add_argument('--min_frac', type=float, default=1e-4, help='The minimum fractional change in pressure to consider')
parser.add_argument('--max_frac', type=float, default=1, help='The maximum fractional change in pressure to consider')

parser.add_argument('--figwidth', type=float, default=8.0, help='The width of the figure')
parser.add_argument('--figheight', type=float, default=8.0, help='The height of the figure')
parser.add_argument('--linewidth', type=float, default=1.0, help='The linewidth of the plot')
parser.add_argument('--cmap_name', type=str, default='Accent', help='The colormap to use')
parser.add_argument('--min_strain', type=float, default=1e-10, help='The minimum strain to plot')
parser.add_argument('--max_strain', type=float, default=1e-5, help='The maximum strain to plot')
parser.add_argument('--fontsize_axis_label', type=float, default=12, help='Axis label fontsize')
parser.add_argument('--fontsize_tick_label', type=float, default=10, help='Tick label fontsize')
parser.add_argument('--fontsize_legend', type=float, default=10, help='Legend fontsize')
parser.add_argument('--fontsize_title', type=float, default=14, help='Title fontsize')

args = parser.parse_args()
lambs = args.lambs
kappa = args.kappa
beta = args.beta
sigma = args.sigma
vp = args.vp
rho = args.rho
max_depth = args.max_depth
num_depths = args.num_depths
temp = args.temp
figwidth = args.figwidth
figheight = args.figheight
linewidth = args.linewidth
cmap_name = args.cmap_name
min_frac = args.min_frac
max_frac = args.max_frac
min_strain = args.min_strain
max_strain = args.max_strain
fontsize_axis_label = args.fontsize_axis_label
fontsize_tick_label = args.fontsize_tick_label
fontsize_legend = args.fontsize_legend
fontsize_title = args.fontsize_title

###
# Constants
###
g = 10.0 # Gravitational acceleration in m/s^2

###
# Compute the thermal elastic stress
###

# Compute the angular frequency and wavenumber
omega = 2 * pi / num_seconds_in_day

# Get the depth array
depths = logspace(0, log10(max_depth), num_depths)
min_depth = depths[0]
max_depth = depths[-1]

# Compute the modulus relating the thermal elastic stress to the thermal elastic strain
modulus = vp ** 2 * rho

# Compute the lithostatic pressure
p_liths = rho * g * depths + bar

# Compute the thermal elastic stress for each lambda
epsilons_xx_dict = {}
epsilons_yy_dict = {}
epsilons_xx_simple_dict = {}
epsilons_yy_simple_dict = {}
frac_dict = {}
frac_simple_dict = {}
for lamb in lambs:
    print(f'Computing the thermal elastic stress for a horizontal wavelength of {lamb} m')
    k = 2 * pi / lamb

    gamma_simple = get_gamma(omega, k, kappa, simplify=True)
    gamma = get_gamma(omega, k, kappa, simplify=False)

    # Compute the two normal thermal elastic strains
    epsilons_simple_xx, epsilons_simple_yy = get_strain(gamma_simple, k, depths, beta, sigma, temp, simplify=True)
    epsilons_xx, epsilons_yy = get_strain(gamma, k, depths, beta, sigma, temp, simplify=False)

    # Compute the pressure perturbation
    p_perturbs = abs(modulus * (epsilons_xx + epsilons_yy))
    p_perturbs_simple = abs(modulus * (epsilons_simple_xx + epsilons_simple_yy))

    # Compute the fractional change in pressure
    frac_p_perturbs = p_perturbs / p_liths
    frac_p_perturbs_simple = p_perturbs_simple / p_liths

    # Store the results
    frac_dict[lamb] = frac_p_perturbs
    frac_simple_dict[lamb] = frac_p_perturbs_simple

    epsilons_xx_dict[lamb] = abs(epsilons_xx)
    epsilons_yy_dict[lamb] = abs(epsilons_yy)

    epsilons_xx_simple_dict[lamb] = abs(epsilons_simple_xx)
    epsilons_yy_simple_dict[lamb] = abs(epsilons_simple_yy)

###
# Plot the thermal elastic strains
###

# Create the figure and axis
fig, ax = subplots(figsize=(figwidth, figheight))

# Get the colormap
cmap = colormaps[cmap_name]

# Plot the results
for i, lamb in enumerate(lambs):
    color = cmap(i)

    # Plot the results
    # epsilons_xx = epsilons_xx_dict[lamb]
    # epsilons_yy = epsilons_yy_dict[lamb]
    epsilons_xx = epsilons_xx_simple_dict[lamb]
    epsilons_yy = epsilons_yy_simple_dict[lamb]
    ax.plot(epsilons_xx, depths, color=color, linewidth=linewidth, linestyle='--', label=f'${lamb:.0f}$ m')
    ax.plot(epsilons_yy, depths, color=color, linewidth=linewidth, linestyle=':')

# Set the axes to log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Set the axes limits
ax.set_xlim(min_strain, max_strain)
ax.set_ylim(min_depth, max_depth)

# Set the axes labels
ax.set_xlabel('Strain', fontsize = fontsize_axis_label)
ax.set_ylabel('Depth (m)', fontsize = fontsize_axis_label)

# Reverse the y-axis
ax.invert_yaxis()

# Turn on the grid
ax.grid(linestyle='--', alpha=0.5, which='both')

# Set the title
ax.set_title(f'Thermoelastic normal strains', 
             fontsize=fontsize_title, fontweight='bold')

# Set the legend
legend = ax.legend(loc='lower right', fontsize=fontsize_legend, frameon=True)
legend.set_title('Wavelength', prop={'size': fontsize_legend, 'weight': 'bold'})

# Save the figure
figname = f'thermoelastic_strains.png'
save_figure(fig, figname)

###
# Plot the fractional change in pressure
###

# Create the figure and axis
fig, ax = subplots(figsize=(figwidth, figheight))

# Get the colormap
cmap = colormaps[cmap_name]

# Plot the results
for i, lamb in enumerate(lambs):
    color = cmap(i)

    # Plot the results
    fracs = frac_dict[lamb]
    fracs_simple = frac_simple_dict[lamb]
    ax.plot(fracs, depths, color=color, linewidth=linewidth, label=f'${lamb:.0f}$ m')
    # ax.plot(fracs_simple, depths, color=color, linewidth=linewidth, linestyle='--')

    # # Plot the thermal boundary layer thickness
    # h = abs(1 / gamma)
    # patch = Rectangle((min_frac, min_depth), max_frac - min_frac, h, color='black', alpha=0.1)
    # ax.add_patch(patch)

# Set the axes to log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Set the axes limits
ax.set_xlim(min_frac, max_frac)
ax.set_ylim(min_depth, max_depth)

# Reverse the y-axis
ax.invert_yaxis()

# Turn on the grid
ax.grid(linestyle='--', alpha=0.5, which='both')

# Set the axes labels
ax.set_xlabel('Fractional change in pressure', fontsize = fontsize_axis_label)
ax.set_ylabel('Depth (m)', fontsize = fontsize_axis_label)

# Set the legend
legend = ax.legend(loc='lower right',
                    fontsize=fontsize_legend, frameon=True)
legend.set_title('Wavelength', prop={'size': fontsize_legend, 'weight': 'bold'})

# Set the title
ax.set_title(f'Thermoelastic-stress induced pressure perturbations', 
             fontsize=fontsize_title, fontweight='bold')

# Save the figure
figname = f'thermoelastic_pressure_perturbations.png'
save_figure(fig, figname)