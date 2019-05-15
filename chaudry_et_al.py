'''Script to create figures for paper


Info
-----
A folder (path_in) with the following structure needs to be generated and the SMIF simulation restuls stored, for 
example in the path (C:/path_results/EW/Central/NF1/SMIF_RESULTS)

... path_in
        EW
            central
                WeatherScenario (i.e. NF1))
                ...
            decentral
                WeatherScenario
                        decentral_step_calculations    # Results for plotting figure 6
                                step1                  # Results for step 1
                                step2                  # Results for step 2
                                step3                  # Results for step 3
                                ...
                ...
        MW     
            ...
'''
import os
import shutil
import chaudry_et_al_functions

# -----------------------------------
# Configure paths
# -----------------------------------
path_out = "C:/_test"   # Path to store results
path_in = "C:/Users/cenv0553/nismod2/results/PLOTTINGFOLDER" # Path with model runs

path_shapefile_energyhub = "C:/Users/cenv0553/plotting_p4/shapefile/EH_Region_Boundaries.shp"
path_shapefile_busbars = "C:/Users/cenv0553/plotting_p4/shapefile/ElectricityBus_29.shp"

# Configure simulation names
simulation_name = 'energy_sd_constrained'   # Name of model

scenarios = ['EW', 'MV'] 
weather_scenario = 'NF1'
fueltype = 'electricity'
unit = 'GW'

# Temporal conversion of results, From seasonal to annual results
factor_from_4_weeks_to_full_year = 1.0 / ((1.0 / (365/24)) * 4)

# ------------------------
# Create empty result folders and delte preivous results
# ------------------------
shutil.rmtree(path_out)
chaudry_et_al_functions.create_folder(path_out)

figs = ['fig3', 'fig4', 'fig5', 'fig6']
modes = ['DECENTRAL', 'CENTRAL']

for fig_name in figs:
    path_fig = os.path.join(path_out, fig_name)
    chaudry_et_al_functions.create_folder(path_fig)

# Add more metric files to generate plots with step wise emission calcultions (also add unit below)
metric_filenames = {
    'emissions_eh': {'output_e_emissions_eh_timestep':'gray'},
    'e_emissions': {'output_e_emissions_timestep': 'green'},
    'h_emissions': {'output_h_emissions_eh_timestep': 'yellow'},

    'gas_load_shed': {'output_gas_load_shed_timestep': 'red'},
    'elec_load_shed': {'output_elec_load_shed_timestep': 'blue'},
    'gas_load_shed_eh': {'output_gas_load_shed_eh_timestep': 'brown'},
    'elec_load_shed_eh': {'output_elec_load_shed_eh_timestep': 'cyan'},

    'tran_wind_curtailed':	{'output_tran_wind_curtailed_timestep': 'pink'},
    'tran_pv_curtailed': {'output_tran_pv_curtailed_timestep': 'orange'},
    
    'elec_cost': {'output_tran_pv_curtailed_timestep': 'magenta'},

    'opt_cost': {'output_total_opt_cost_timestep': 'green'},
    }

# Units of metric
unit_metric = {
    'emissions_eh': 'set_unit',
    'e_emissions': 'set_unit',
    'h_emissions': 'set_unit',
    
    'gas_load_shed': 'set_unit',
    'elec_load_shed': 'set_unit',
    'gas_load_shed_eh': 'set_unit',
    'elec_load_shed_eh': 'set_unit',

    'tran_wind_curtailed': 'set_unit',
    'tran_pv_curtailed': 'set_unit',

    'elec_cost': 'set_unit',

    'total_opt_cost': 'set_unit',
}

# For plotting maps

# Files to plot for xy-plot and pie-chart (p4)
filenames = {

    #Electricity Generation mix in energy hubs
    'elec_hubs': { 
        'output_eh_gas_fired_other_timestep': 'gray',
        'output_eh_chp_gas_timestep': 'fuchsia',
        'output_eh_chp_biomass_timestep': 'green',
        'output_eh_chp_waste_timestep': 'violet',
        'output_eh_fuel_cell_timestep': 'slateblue',
        'output_eh_wind_curtalied_timestep': 'firebrick',
        'output_eh_wind_power_timestep': 'blue',
        'output_eh_tran_e_timestep': 'orangered',
        'output_eh_tran_e_export_timestep': 'red'
        }, 

    #Electricity Generation mix in electricity transmisison
    'elec_transmission': {
        'output_tran_gas_fired_timestep': 'olivedrab',
        'output_tran_coal_timestep': 'olivedrab',
        'output_tran_pump_power_timestep': 'olivedrab',
        'output_tran_hydro_timestep': 'olivedrab',
        'output_tran_nuclear_timestep': 'olivedrab',
        'output_tran_interconnector_timestep': 'olivedrab',
        'output_tran_renewable_timestep': 'olivedrab',
        'output_e_reserve_timestep': 'olivedrab',
        'output_tran_wind_offshore_timestep': 'olivedrab',
        'output_tran_wind_onsore_timestep': 'olivedrab',
        'output_tran_pv_power_timestep': 'olivedrab',
        'output_tran_pv_curtailed_timestep': 'olivedrab'},

    #Heat Supply Mix in energy hubs
    'heat_hubs':{ 
        'output_eh_gasboiler_b_timestep': 'darkcyan',
        'output_eh_heatpump_b_timestep': 'royalblue',
        'output_eh_gasboiler_dh_timestep': 'orange',
        'output_eh_gaschp_dh_timestep': 'y',
        'output_eh_heatpump_dh_timestep': 'indianred',
        'output_eh_biomassboiler_b_timestep': 'red',
        'output_eh_biomassboiler_dh_timestep': 'gold',
        'output_eh_biomasschp_dh_timestep': 'darkgreen',
        'output_eh_wastechp_dh_timestep': 'darkmagenta',
        'output_eh_electricboiler_b_timestep': 'aqua',
        'output_eh_electricboiler_dh_timestep': 'grey',
        'output_eh_hydrogenboiler_b_timestep': 'magenta',
        'output_eh_hydrogen_fuelcell_dh_timestep': 'brown',
        'output_eh_hydrogen_heatpump_b_timestep': 'olive'},

    #Natural gas supply mix in gas transmissios
    'gas_transmission': {
        'output_gas_domestic_timestep': 'olivedrab',
        'output_gas_lng_timestep': 'olivedrab',
        'output_gas_interconnector_timestep': 'olivedrab',
        'output_gas_storage_timestep': 'olivedrab',
        'output_storage_level_timestep': 'olivedrab',
    },
    #Load Shedding (Unserved demand)
    'load_shedding': {
        'output_gas_load_shed_timestep': 'olivedrab',
        'output_elec_load_shed_timestep': 'olivedrab',
        'output_gas_load_shed_eh_timestep': 'olivedrab',
        'output_elec_load_shed_eh_timestep': 'olivedrab',
    },
    #Gas Supply Mix in Energy hubs
    'gas_hubs': {
        'output_eh_tran_g_timestep': 'olivedrab',
        'output_eh_gas_qs_timestep': 'olivedrab',
        'output_eh_gstorage_level_timestep': 'olivedrab',
    },
    #Hydrogen supply mix in energy hubs
    'hydrogen_hubs': {
        'output_eh_h2_timestep': 'olivedrab',
        'output_eh_h2_qs_timestep': 'olivedrab',
        'output_eh_h2storage_level_timestep': 'olivedrab',
    },
    #Output oil
    'oil_output': {
        'output_oil_demand_heat_timestep': 'olivedrab',
        'output_oil_demand_timestep': 'olivedrab'
    },
    #Output oil
    'solidfuel_output': {
        'output_solidfuel_demand_heat_timestep': 'olivedrab',
        'output_solidfuel_demand_timestep': 'olivedrab'
    }
}
filenames_fueltypes = {
    'electricity': [],
    'gas': [],
    'hydrogen': [],
}

# Provide x-value limites
x_values_lims = {
    'elec_hubs': {'nr_of_bins': 2, 'bin_value': 10},
    'heat_hubs': {'nr_of_bins': 3, 'bin_value': 10},
}


steps = ['step1', 'step2', 'step3'] #, 'step4']
years = [2015, 2030, 2050]

# ------------------------
# Load data
# ------------------------
data_container, data_container_fig_steps = chaudry_et_al_functions.load_data(
    path_in,
    simulation_name=simulation_name,
    scenarios=scenarios,
    unit=unit,
    steps=steps,
    modes=modes)

print("... finished loading data", flush=True)

# ------------------------
# Create figures
# ------------------------

print("plotting Figs 3 and 4", flush=True)
chaudry_et_al_functions.plot_figures(
    path_in,
    path_out,
    data_container,
    filenames=filenames,
    filenames_fueltypes=filenames_fueltypes,
    scenarios=scenarios,
    weather_scearnio=weather_scenario,
    types_to_plot=['elec_hubs', 'heat_hubs'],
    unit=unit,
    x_values_lims=x_values_lims,
    modes=modes,
    temporal_conversion_factor=factor_from_4_weeks_to_full_year,
    years=years)

print("... start plotting Fig6", flush=True)
chaudry_et_al_functions.plot_maps(
    path_out,
    path_shapefile_energyhub,
    path_shapefile_busbars,
    data_container,
    metric_filenames=metric_filenames,
    years=years,
    scenarios=scenarios,
    weather_scearnio=weather_scenario,
    modes=modes,
    temporal_conversion_factor=factor_from_4_weeks_to_full_year,
    create_cartopy_maps=True)

print("... plotted Fig 6", flush=True)

try:
    chaudry_et_al_functions.plot_step_figures(
            path_out=path_out,
            data_container_fig_steps=data_container_fig_steps,
            metric_filenames=metric_filenames,
            scenarios=scenarios,
            weather_scearnio=weather_scenario,
            steps=steps,
            unit_metric=unit_metric,
            temporal_conversion_factor=factor_from_4_weeks_to_full_year,
            years=years)
    print("... plotted Fig5", flush=True)
except:
    print("could not create step figures")


print("... Finished creasting figures", flush=True)