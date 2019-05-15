"""Functions to generate figures
"""
import os
import math
from collections import OrderedDict
import pandas as pd
import numpy as np
from collections import defaultdict
from tabulate import tabulate
import argparse

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colorbar, colors
from matplotlib.colors import Normalize

# Set default font
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
#rcParams.update({'figure.autolayout': True})

def clear_figure(plt, fig, ax):
    """Clear figure contents and axis
    """
    try:
        plt.close('all')
    except:
        pass
    try:
        fig.clear()
    except:
        pass
    try:
        ax.remove()
    except:
        pass

def autolabel(rects, ax, rounding_digits, value_show_crit=0.01):
    """Place labels with percentages in middle of stacked plots
    """
    for rect in rects:
        value = rect.get_width()
        value_round = round(value, rounding_digits)

        if value_round > value_show_crit:
            ax.text(rect.get_x() + rect.get_width() / 2,#width
                    rect.get_y() + rect.get_height() / 2.4, #height
                    s=value_round,
                    ha='center',
                    va='center',
                    color="White",
                    fontsize=6,
                    fontweight="bold")

def cm2inch(*tupl):
    """Convert input cm to inches (width, hight)
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def write_list_to_txt(path_result, list_out):
    """Write scenario population for a year to txt file
    """
    file = open(path_result, "w")
    for entry in list_out:
        file.write(entry + "\n")
    file.close()

def write_to_txt(path_result, entry):
    """Write scenario population for a year to txt file
    """
    file = open(path_result, "w")
    file.write(entry + "\n")
    file.close()

def export_legend(legend, filename="legend.png"):
    """Export legend as seperate file
    """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def create_folder(path_folder, name_subfolder=None):
    """Creates folder or subfolder

    Arguments
    ----------
    path : str
        Path to folder
    folder_name : str, default=None
        Name of subfolder to create
    """
    if not name_subfolder:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
    else:
        path_result_subolder = os.path.join(path_folder, name_subfolder)
        if not os.path.exists(path_result_subolder):
            os.makedirs(path_result_subolder)

def load_data(in_path, scenarios, simulation_name, unit, steps, modes):
    """Read results and set timestep as index in dataframe

    Returns
    data_container : dataframes in [scenario][mode][weather_scenario]
    """
    plot_decentarl_figures = True
    data_container = {}
    data_container_fig_steps = {}

    for scenario in scenarios:
        data_container[scenario] = {}
        data_container_fig_steps[scenario] = {}

        for mode in modes:
            data_container[scenario][mode] = {}
            data_container_fig_steps[scenario][mode]  = {}

            # Iterate over weather scenarios
            path_scenarios = os.path.join(in_path, scenario, mode)
            weather_scenarios = os.listdir(path_scenarios)

            for weather_scenario in weather_scenarios:
                data_container[scenario][mode][weather_scenario] = {}
                data_container_fig_steps[scenario][mode][weather_scenario] = {}

                # ---------------------------
                # Load restuls for fig6
                # ---------------------------
                try:
                    if mode == 'DECENTRAL':

                        for step in steps:
                            data_container_fig_steps[scenario][mode][weather_scenario][step] = {}
                            data_container_fig_steps[scenario][mode][weather_scenario][step]['energy_supply_constrained'] = {}
                            
                            path_supply_mix = os.path.join(in_path, scenario, mode, weather_scenario, 'decentral_step_calculations', step, simulation_name, 'energy_supply_constrained', 'decision_0')

                            all_files = os.listdir(path_supply_mix)

                            for file_name in all_files:
                                path_file = os.path.join(path_supply_mix, file_name)
                                variable_name = file_name[7:-18]
                                data_file = pd.read_csv(path_file)

                                try: # test if seasonal_week_attribute
                                    _ = data_file.groupby(data_file['seasonal_week']).sum()
                                    data = data_file.set_index('seasonal_week')

                                    if unit == 'GW':
                                        data[variable_name] = data[variable_name] / 1000.0
                                    if unit == 'MW':
                                        pass
                                except:
                                    print("{} Data containes no seasonal_week attribute".format(file_name))

                                data_container_fig_steps[scenario][mode][weather_scenario][step]['energy_supply_constrained'][file_name] = data
                except:
                    print("ERROR: could not read step data")

                # ---------------------------
                # Load energy_supply_constrained
                # ---------------------------
                data_container[scenario][mode][weather_scenario]['energy_supply_constrained'] = {}
                path_supply_mix = os.path.join(in_path, scenario, mode, weather_scenario, simulation_name, 'energy_supply_constrained', 'decision_0')

                all_files = os.listdir(path_supply_mix)

                for file_name in all_files:
                    path_file = os.path.join(path_supply_mix, file_name)
                    variable_name = file_name[7:-18]
                    data_file = pd.read_csv(path_file)

                    try:
                        # test if seasonal_week_attribute
                        _ = data_file.groupby(data_file['seasonal_week']).sum()
                        data = data_file.set_index('seasonal_week')

                        if unit == 'GW':
                            data[variable_name] = data[variable_name] / 1000.0
                        if unit == 'MW':
                            pass
                    except:
                        print("{} Data containes no seasonal_week attribute".format(file_name))

                    data_container[scenario][mode][weather_scenario]['energy_supply_constrained'][file_name] = data

    return data_container, data_container_fig_steps

'''def choropleth(ax, attr, cmap_name): #
    # We need to normalize the values before we can
    # use the colormap.
    values = [c.attributes[attr] for c in africa]
    norm = Normalize(
        vmin=min(values), vmax=max(values))
    cmap = plt.cm.get_cmap(cmap_name)
    for c in africa:
        v = c.attributes[attr]
        sp = ShapelyFeature(c.geometry, crs,
                            edgecolor='k',
                            facecolor=cmap(norm(v)))
        ax.add_feature(sp)
'''
'''
def classify_values(diff_2020_2050_reg_share):      
    """Classify values
    """
    regions = diff_2020_2050_reg_share.index
    relclassified_values = pd.DataFrame()
    relclassified_values['reclassified'] = 0
    relclassified_values['name'] = regions
    relclassified_values = relclassified_values.set_index(('name'))
    relclassified_values['name'] = regions

    for region in regions.values:
        relclassified_values.loc[region, 'reclassified'] = clasify_color(diff_mean, diff_std, threshold=threshold)

    return relclassified_values
'''
'''
def clasify_color(diff_mean, diff_std, threshold):
    if diff_mean < -1 * threshold and diff_std < -1 * threshold:
        color_pos = 3
    elif diff_mean > threshold and diff_std < -1 * threshold:
        color_pos = 2
    elif diff_mean < -1 * threshold and diff_std > threshold:
        color_pos = 4
    elif diff_mean > threshold and diff_std > threshold:
        color_pos = 1
    else:
        color_pos = 0

    return color_pos 
'''
def plot_maps(
        path_out,
        path_shapefile_energyhub,
        path_shapefile_busbars,
        data_container,
        metric_filenames,
        scenarios,
        years,
        weather_scearnio,
        modes,
        temporal_conversion_factor,
        seperate_legend=True,
        create_cartopy_maps=True

    ):
    """Plot spatial maps

    Choropleth example: https://ipython-books.github.io/146-manipulating-geospatial-data-with-cartopy/
    """
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader

    path_out_folder_fig6 = os.path.join(path_out, 'fig6')

    # Read shapefiles
    shapefile_energyhub = shpreader.Reader(path_shapefile_energyhub)
    shapefile_bus_bars = shpreader.Reader(path_shapefile_busbars)
    
    regions_energyhub = []
    for record in shapefile_energyhub.records():
        region_name = record.attributes['Region_Num']
        regions_energyhub.append(region_name)

    regions_busbars = []
    for record in shapefile_bus_bars.records():
        region_name = record.attributes['Bus_No']
        regions_busbars.append(region_name)


    fig_dict = {}
    for year in years:
        fig_dict[year] = {}

        for scenario in scenarios:
            for metric, filenames in metric_filenames.items():

                # Create busbars and regions empty dataframes
                df_to_plot_energyhubs = pd.DataFrame(regions_energyhub, columns=['regions'])
                df_to_plot_energyhubs = df_to_plot_energyhubs.set_index('regions')
                df_to_plot_national = pd.DataFrame(columns=['national'])

                df_to_plot_busbars = pd.DataFrame(regions_busbars, columns=['regions'])
                df_to_plot_busbars = df_to_plot_busbars.set_index('regions')

                for mode in modes:
                    #print("======{}  {} {}".format(year, scenario, mode))
                    for metric_file_name, color_metric in filenames.items():
                        #print("metric_file_name: " + str(metric_file_name), flush=True)
                        df_to_plot_energyhubs[mode] = 0
                        df_to_plot_busbars[mode] = 0
                        df_to_plot_national[mode] = 0

                        data_files = data_container[scenario][mode][weather_scearnio]['energy_supply_constrained']

                        for file_name, file_data in data_files.items():

                            # Aggregate regional  
                            try: 
                                regional_annual = file_data.set_index('energy_hub')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                                region_type = 'energy_hub'
                            except:
                                pass

                            try:
                                regional_annual = file_data.set_index('bus_bars')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                                region_type = 'bus_bars'
                            except:
                                pass
                
                            try:
                                regional_annual = file_data.set_index('gas_nodes')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                                region_type = 'gas_nodes'
                            except:
                                pass

                            try:
                                regional_annual = file_data.set_index('national')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                                region_type = 'national'
                            except:
                                pass

                            if region_type == 'gas_nodes':
                                # Skip
                                pass
                            else:
                                file_name_split_no_timpestep = file_name[:-9]    #remove ending
                                name_column = file_name_split_no_timpestep[7:-9] #remove output_ and ending
                                file_name_split = file_name.split("_")
                                year_simulation = int(file_name_split[-1][:4])

                                if year == year_simulation:
                                    if file_name_split_no_timpestep == metric_file_name:
                                        region_type_metric = region_type

                                        if region_type == 'bus_bars':
                                            df_to_plot_busbars[mode] = regional_annual[name_column].tolist()
                                        if region_type == 'energy_hub':
                                            df_to_plot_energyhubs[mode] = regional_annual[name_column].tolist()
                                        if region_type == 'national':
                                            df_to_plot_national[mode] = regional_annual.sum()

                                        # Write out all results (that could be plotted in ArcGIS)
                                        if region_type == 'bus_bars':
                                            values = df_to_plot_busbars[mode].to_frame()
                                        if region_type == 'energy_hub':
                                            values = df_to_plot_energyhubs[mode].to_frame()
                                        if region_type == 'national':
                                            values = regional_annual

                                        filename = "{}_{}_{}_{}_{}__map_values.txt".format(year, scenario, metric, mode, region_type)
                                        values.to_csv(
                                            os.path.join(path_out_folder_fig6, filename),
                                            header=True, index=True, sep=',')
                
                if region_type_metric == 'bus_bars':
                    fig_dict[year][metric] = {'region_type': region_type_metric, 'data': df_to_plot_busbars}
                if region_type_metric == 'energy_hub':
                    fig_dict[year][metric] = {'region_type': region_type_metric, 'data': df_to_plot_energyhubs}
                if region_type_metric == 'national':
                    fig_dict[year][metric] = {'region_type': region_type_metric, 'data': []}

    if create_cartopy_maps:
        # Actual plotting with cartopy

        # Classification colors
        cmap_name = 'Reds'

        color_manuals = {
            0: '#f7f7f7', #Threshold change limit
            1: '#e66101', #'red',
            2: '#fdb863', #'tomato',
            3: '#5e3c99', #'seagreen',
            4: '#b2abd2'} #'orange'}

        for year in years:
            for metric in metric_filenames.keys():
                for mode in modes:
                    
                    region_type = fig_dict[year][metric]['region_type']
                    #print("=====fff {}  {}  {} {} ".format(year, metric, mode, region_type))
                    if region_type == 'national':
                        pass
                    else:
                        values = fig_dict[year][metric]['data'][mode]

                        # Choropleth map colors, colorbar colors
                        min_value = min(values)
                        max_value = max(values)
                        norm = Normalize(vmin=min_value, vmax=max_value)
                        cmap = plt.cm.get_cmap(cmap_name)

                        # Use Cartopy to plot geometries with reclassified faceolor
                        plt.figure(figsize=cm2inch(4, 6), dpi=300)
                        proj = ccrs.OSGB()
                        ax = plt.axes(projection=proj)
                        ax.outline_patch.set_visible(False)

                        # set up a dict to hold geometries keyed by our key
                        geoms_by_key = defaultdict(list)
                        
                        # for each records, pick out our key's value from the record
                        # and store the geometry in the relevant list under geoms_by_key
                        if region_type == 'bus_bars':
                            for record in shapefile_bus_bars.records():
                                region_name = record.attributes['Bus_No']
                                geoms_by_key[region_name].append(record.geometry)
                        if region_type == 'energy_hub':
                            for record in shapefile_energyhub.records():
                                region_name = record.attributes['Region_Num']
                                geoms_by_key[region_name].append(record.geometry)

                        # now we have all the geometries in lists for each value of our key
                        # add them to the axis, using the relevant color as facecolor
                        for region_name, geoms in geoms_by_key.items():
                            value_region = values[region_name]

                            # Manual classification (see fig_3_plot_over_time)
                            #region_reclassified_value = reclassified.loc[key]['reclassified']
                            #facecolor = color_manuals[region_reclassified_value]

                            # Choropleth
                            facecolor = cmap(norm(value_region))
                            ax.add_geometries(geoms, crs=proj, edgecolor='black', facecolor=facecolor, linewidth=0.1)

                        filname = "{}_{}_{}_{}__map".format(year, scenario, metric, mode)

                        # Colorbar
                        if seperate_legend: 
                            pass
                        else:
                            # Add legend to figure
                            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            sm._A = []
                            plt.colorbar(sm, ax=ax)

                        # Save figure
                        plt.tight_layout()
                        plt.savefig(os.path.join(path_out_folder_fig6, "{}.pdf".format(filname)))
                        plt.close()

                        # Save colorbar
                        if seperate_legend:
                            # draw a new figure and replot the colorbar there
                            fig_cb, ax_cb = plt.subplots(figsize=cm2inch(1, 3))
                            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            sm._A = []
                            plt.colorbar(sm, ax=ax_cb)
                            plt.axis('off')

                            plt.savefig(
                                os.path.join(path_out_folder_fig6, "{}__legend.pdf".format(filname)),
                                bbox_inches='tight')
                            
                            plt.close()

def plot_step_figures(
        path_out,
        data_container_fig_steps,
        metric_filenames,
        scenarios,
        weather_scearnio,
        steps,
        unit_metric,
        temporal_conversion_factor,
        years=[2015, 2030, 2050],
        seperate_legend=True
    ):
    """Plot decentral step results
    """
    fig_dict = {}
    path_out_folder_fig5 = os.path.join(path_out, 'fig5')
    mode = 'DECENTRAL'

    color_scenarios = {
        'MV': 'brown',
        'EW': 'steelblue'}

    for year in years:
        fig_dict[year] = {}
        
        for metric, filenames in metric_filenames.items():
            fig_dict[year][metric] = {}
            colors = []
            df_to_plot = pd.DataFrame(steps, columns=['x_labels'])
            df_to_plot = df_to_plot.set_index('x_labels')

            for scenario in scenarios:
                colors.append(color_scenarios[scenario])
                df_to_plot[scenario] = 0 #fill with empty

                for metric_file_name, color_metric in filenames.items():

                    for step in steps:
                        data_files = data_container_fig_steps[scenario][mode][weather_scearnio][step]['energy_supply_constrained']

                        for file_name, file_data in data_files.items():

                            # Aggregate national data for every timesteps
                            ##national_per_timesteps = file_data.groupby(file_data.index).sum()

                            # Aggregate regional annual data
                            try:
                                regional_annual = file_data.set_index('energy_hub')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                            except:
                                #print("no energy_hub attribute")
                                pass
                            try:
                                regional_annual = file_data.set_index('bus_bars')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                            except:
                                #print("no 'bus_bars'")
                                pass
                            try:
                                regional_annual = file_data.set_index('gas_nodes')
                                regional_annual = regional_annual.groupby(regional_annual.index).sum()
                            except:
                                #print("no gas_nodes attribute")
                                pass

                            file_name_split_no_timpestep = file_name[:-9] #remove ending
                            name_column = file_name_split_no_timpestep[7:-9] #remove output_ and ending
                            file_name_split = file_name.split("_")
                            year_simulation = int(file_name_split[-1][:4])

                            if year == year_simulation:
                                if file_name_split_no_timpestep == metric_file_name:
                                    #df_to_plot[scenario][step] = national_per_timesteps[name_column]
                                    df_to_plot[scenario][step] = np.sum(regional_annual[name_column]) # Add National annual

            fig_dict[year][metric] = df_to_plot

        # ------------------------------------
        # Plot metrics
        # ------------------------------------
        for year, metrics in fig_dict.items():
            for metric, scenario_data in metrics.items():
                table_all_regs = []

                #data_scenario_steps = df_to_plot
                df_to_plot = scenario_data

                fig, ax = plt.subplots()

                # Plot lines
                df_to_plot.plot(
                    kind='line',
                    ax=ax,
                    colors=colors)

                # Plot line dots
                df_to_plot.plot(
                    style=['.', '^'],
                    ax=ax,
                    ms=6,
                    colors=colors,
                    clip_on=False)
                
                table_all_regs = []
                headers = df_to_plot.columns.tolist()
                headers.insert(0, "step")
                for i in df_to_plot.index:
                    step_values = df_to_plot.loc[i].tolist()
                    step_values.insert(0, i)
                    table_all_regs.append(step_values)

                # Legend
                # ------------
                handles, labels = plt.gca().get_legend_handles_labels()

                by_label = OrderedDict(zip(labels, handles))
                legend = plt.legend(
                    by_label.values(),
                    by_label.keys(),
                    ncol=2,
                    prop={'size': 8},
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.1),
                    frameon=False)

                # Remove frame
                # ------------
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)

                # ------------------
                # Ticks and labels
                # ------------------
                plt.tick_params(
                    axis='y',
                    which='both',
                    left=True,
                    right=False,
                    bottom=False,
                    top=False,
                    labelbottom=False)
                
                # Remove minor ticks x-axis and add again
                #plt.tick_params(axis='x', which='major', bottom=False)
                #ax.set_xticklabels(df_to_plot.index.tolist())
                
                ticks = range(len(steps))
                labels = df_to_plot.index.tolist()
                plt.xticks(
                    ticks=ticks,
                    labels=labels,
                    fontsize=8)
    
                # Limits
                plt.xlim(-1, len(steps))

                #Axis label
                ax.set_xlabel('')
                ax.set_ylabel('{} [{}]'.format(metric, unit_metric[metric]))

                # Reset figure size
                fig = plt.gcf()
                fig.set_size_inches(cm2inch(6, 6))

                fig_name = "{}_{}__metric_plot.pdf".format(metric, year)
                path_out_file = os.path.join(path_out_folder_fig5, fig_name)
                seperate_legend = True
                if seperate_legend:
                    export_legend(
                        legend,
                        os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                    legend.remove()

                plt.savefig(path_out_file, transparent=True, bbox_inches='tight')

                # Write out results to txt
                table_all_regs_tabulate = tabulate(
                    table_all_regs,
                    headers=headers,
                    numalign="right")
                write_to_txt(path_out_file[:-4] + ".txt", table_all_regs_tabulate)

    return

def plot_figures(
        path_in,
        path_out,
        data_container,
        filenames,
        filenames_fueltypes,
        scenarios,
        weather_scearnio,
        types_to_plot,
        unit,
        x_values_lims,
        modes,
        temporal_conversion_factor,
        years=[2015, 2030, 2050],
        seperate_legend=True
    ):
    """Create x-y chart of a time-span (x-axis: demand, y-axis: time)
    """
    # Configuration
    fontsize_small = 8
    fontsize_large = 10
    annote_crit = False #Add labels

    left = 'CENTRAL'
    right = 'DECENTRAL'

    # Font info axis labels
    font_additional_info = {
        'color': 'black',
        'weight': 'bold',
        'size': fontsize_large}

    # https://www.ofgem.gov.uk/data-portal/electricity-generation-mix-quarter-and-fuel-source-gb
    #Colors partly from ofgem
    fueltypes_coloring = {
        'biomass': '#333333',
        'electricity': '#FFCC33',
        'gas': '#6699CC',
        'hydrogen': '#CC6666',
        'oil': '#339966',
        'solidfuel': '#FF9966',
        'waste': '#33CCCC'}

    colors_right_left = {
        right: 'orange',
        left: 'royalblue'}

    for fueltype in types_to_plot:
        print(".... fueltype: {}".format(fueltype))

        # Select hours to plots
        height_cm_xy_figure = 5 # Figure height of xy figure

        seasonal_week_day = 2

        # Select hours from 627 hours from supply model outputs (index 1 to 627)
        hours_selected = range(24 * (seasonal_week_day) + 1, 24 * (seasonal_week_day + 1) + 1)
        hours_selected = range(35, 59)

        # Select a full week
        #hours_selected = range(24 * (0) + 1, 24 * (6 + 1) + 1)

        fig_dict = {}
        fig_dict_piecharts = {}
        fig_dict_regional_annual_demand = {}

        path_out_folder_fig3 = os.path.join(path_out, 'fig3')
        path_out_folder_fig4 = os.path.join(path_out, 'fig4')

        # ========================================================
        # Fueltype chart showing the split between fueltypes
        # ========================================================
        for scenario in scenarios:
            table_out = []
            for mode in [right, left]:

                #data_fueltypes = fig_dict_fuelypes[year][mode][scenario]
                filename_all = "{}_{}_demand_all_byfuel.csv".format(scenario, mode)
                filename_heating = "{}_{}_demand_heating_byfuel.csv".format(scenario, mode)
                path_filename_all = os.path.join(path_in, filename_all)
                path_heating = os.path.join(path_in, filename_heating)
                
                df_all = pd.read_csv(path_filename_all)
                df_heating = pd.read_csv(path_heating)

                for data_type in [df_all, df_heating]:
                    for year in years:
                        fig, ax = plt.subplots()
                        
                        data_fueltypes_p = data_type[data_type['year'] == year]

                        data_fueltypes_p = data_fueltypes_p.round(1)
                        data_fueltypes_p = data_fueltypes_p.set_index('parameter')
                        data_fueltypes_p = data_fueltypes_p.drop(columns=['year'])

                        data_fueltypes_p = data_fueltypes_p.T #transpose
                        ax = data_fueltypes_p.plot(
                            kind='barh',
                            stacked=True,
                            width=0.7,
                            color=fueltypes_coloring.values())

                        # Position labels
                        autolabel(ax.patches, ax, rounding_digits=3)

                        # ------------
                        handles, labels = plt.gca().get_legend_handles_labels()

                        by_label = OrderedDict(zip(labels, handles)) # remove duplicates
                        legend = plt.legend(
                            by_label.values(),
                            by_label.keys(),
                            ncol=2,
                            prop={'size': 10},
                            loc='upper center',
                            bbox_to_anchor=(0.5, -0.1),
                            frameon=False)

                        # Empty y ticks
                        plt.yticks(ticks=[0], labels=[''], fontsize=fontsize_small)

                        # Remove ticks
                        plt.tick_params(axis='x', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
                        plt.tick_params(axis='y', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)

                        #Axis label
                        ax.set_xlabel('')
                        ax.set_ylabel('')

                        # Save pdf of figure and legend
                        # ------------
                        fig_name = "{}_{}_{}_{}__fueltypes_p.pdf".format(scenario, year, fueltype, mode)
                        path_out_file = os.path.join(path_out_folder_fig3, fig_name)

                        if seperate_legend:
                            export_legend(
                                legend,
                                os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                            legend.remove()

                        # Limits
                        # ------------
                        plt.autoscale(enable=True, axis='x', tight=True)
                        plt.autoscale(enable=True, axis='y', tight=True)
                        plt.tight_layout()

                        # Reset figure size
                        fig = plt.gcf()
                        fig.set_size_inches(cm2inch(4.0, 0.3))

                        # Remove frame
                        # ------------
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)

                        plt.savefig(path_out_file)
                        clear_figure(plt, fig, ax)

        # Collect data
        for year in years:
            fig_dict[year] = {}
            fig_dict_piecharts[year] = {}
            fig_dict_regional_annual_demand[year] = {}

            for mode in modes:
                fig_dict[year][mode] = {}
                fig_dict_piecharts[year][mode] = {}
                fig_dict_regional_annual_demand[year][mode] = {}

                for scenario in scenarios:
                    fig_dict[year][mode][scenario] = {}
                    colors_xy_plot = []
                    data_files = data_container[scenario][mode][weather_scearnio]['energy_supply_constrained']

                    files_to_plot = filenames[fueltype].keys()

                    # Get all correct data to plot
                    df_to_plot = pd.DataFrame()
                    df_to_plot_regional = pd.DataFrame()

                    for file_name, file_data in data_files.items():

                        # Aggregate national data for every timesteps
                        national_per_timesteps = file_data.groupby(file_data.index).sum()

                        # Aggregate regional annual data
                        try:
                            regional_annual = file_data.set_index('energy_hub')
                            regional_annual = regional_annual.groupby(regional_annual.index).sum()
                        except:
                            pass

                        file_name_split_no_timpestep = file_name[:-9] #remove ending
                        name_column = file_name_split_no_timpestep[7:-9] #remove output_ and ending
                        file_name_split = file_name.split("_")
                        year_simulation = int(file_name_split[-1][:4])

                        if year == year_simulation:
                            if file_name_split_no_timpestep in files_to_plot:
    
                                # Add national_per_timesteps
                                df_to_plot[str(name_column)] = national_per_timesteps[name_column]
                                df_to_plot_regional[str(name_column)] = regional_annual[name_column]

                                color = filenames[fueltype][file_name_split_no_timpestep]
                                colors_xy_plot.append(color)

                    # Aggregate annual demand for pie-charts
                    fig_dict_piecharts[year][mode][scenario] = df_to_plot.sum()
                    fig_dict_regional_annual_demand[year][mode][scenario] = df_to_plot_regional.sum(axis=1)
                    fig_dict[year][mode][scenario] = df_to_plot.loc[hours_selected]

            # ------------------------------------
            # PLotting regional scpecific bar charts
            # ------------------------------------
            for scenario in scenarios:
                table_all_regs = []

                # Data and plot
                # ------------
                regions_right = fig_dict_regional_annual_demand[year][right][scenario]
                regions_left = fig_dict_regional_annual_demand[year][left][scenario]

                # Sorting
                sored_df = regions_right.sort_values()
                sorted_index = sored_df.index.tolist()

                # Reorder index
                regions_right = regions_right.reindex(sorted_index)
                regions_left = regions_left.reindex(sorted_index)

                # Convert from GW to TW
                regions_right = regions_right / 1000
                regions_left = regions_left / 1000
                
                # -------------------------
                # Plotting all regions together
                # -------------------------
                fig, ax = plt.subplots()

                df_bars = pd.DataFrame(
                    {right: regions_right.values.tolist(),
                    left: regions_left.values.tolist()},
                    index=regions_right.index)
                
                # Writing out txt
                headers_all_regs = df_bars.columns.values.tolist()
                headers_all_regs.insert(0, 'energy_hubs')
                for index in df_bars.index:
                    reg_val = df_bars.loc[index].values.tolist()
                    reg_val.insert(0, index)
                    table_all_regs.append(reg_val)
    
                ax = df_bars.plot(
                    kind='bar',
                    width=0.8,
                    color=list(colors_right_left.values()))

                ax.grid(which='major', color='white', axis='y', linestyle='-')

                # Legend
                # ------------
                handles, labels = plt.gca().get_legend_handles_labels()

                by_label = OrderedDict(zip(labels, handles))
                legend = plt.legend(
                    by_label.values(),
                    by_label.keys(),
                    ncol=2,
                    prop={'size': 8},
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.1),
                    frameon=False)

                # Remove frame
                # ------------
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(False)
    
                # ------------------
                # Ticks and labels
                # ------------------
                plt.tick_params(axis='y', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
    
                #Axis label
                ax.set_xlabel('Energy hub region')
                ax.set_ylabel('TW')

                # Reset figure size
                fig = plt.gcf()
                fig.set_size_inches(cm2inch(12, 6))

                fig_name = "{}_{}_{}__barplots_comparison_all.pdf".format(scenario, year, fueltype)
                path_out_file = os.path.join(path_out_folder_fig4, fig_name)
                seperate_legend = True
                if seperate_legend:
                    export_legend(
                        legend,
                        os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                    legend.remove()

                plt.savefig(path_out_file, transparent=True, bbox_inches='tight')
                clear_figure(plt, fig, ax)

                # Write out results to txt
                table_all_regs_tabulate = tabulate(
                    table_all_regs,
                    headers=headers_all_regs,
                    numalign="right")
                write_to_txt(path_out_file[:-4] + ".txt", table_all_regs_tabulate)

                # ------------------------
                # Every region on its own
                # ------------------------
                table_out = []
                for region in regions_right.index:
                    tot_right = regions_right.loc[region]
                    tot_left = regions_left.loc[region]

                    df_bars = pd.DataFrame([[tot_right, tot_left]], columns=[right, left])
                    headers = list(df_bars.columns)
                    headers.insert(0, "hour")
                    headers.insert(0, "type")

                    for index_hour in df_bars.index:
                        row = list(df_bars.loc[index_hour])
                        row.insert(0, index_hour)
                        row.insert(0, 'right')
                        table_out.append(row)
                    
                    fig, ax = plt.subplots()

                    ax = df_bars.plot(
                        kind='bar',
                        y=df_bars.columns.tolist(),
                        color=list(colors_right_left.values()),
                        width=0.4)
  
                    # Legend
                    # ------------
                    handles, labels = plt.gca().get_legend_handles_labels()

                    by_label = OrderedDict(zip(labels, handles))
                    legend = plt.legend(
                        by_label.values(),
                        by_label.keys(),
                        ncol=2,
                        prop={'size': 8},
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        frameon=False)

                    # Remove frame
                    # ------------
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(True)
                    ax.spines['left'].set_visible(False)

                    # Add grid lines
                    # ------------
                    plt.tick_params(axis='y', which='both', left=False) #remove ticks
                    
                    # ------------------
                    # Ticks and labels
                    # ------------------
                    interval = 1
                    nr_of_intervals = 2
                    max_tick = (nr_of_intervals * interval)
                    ticks = [round(i * interval,2)  for i in range(nr_of_intervals)]
                    labels = [str(round(i * interval, 2)) for i in range(nr_of_intervals)]
    
                    plt.yticks(
                        ticks=ticks,
                        labels=labels,
                        fontsize=8)

                    # ------------
                    # Limits
                    # ------------
                    plt.ylim(0, max_tick)

                    # Remove ticks and labels
                    plt.tick_params(axis='x', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
                    plt.tick_params(axis='y', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])

                    #Axis label
                    ax.set_xlabel('')
                    ax.set_ylabel('')

                    # Reset figure size
                    widht=0.5
                    height=3
                    fig = plt.gcf()
                    fig.set_size_inches(cm2inch(widht, height))

                    # Save pdf of figure and legend
                    # ------------
                    fig_name = "{}_{}_{}__{}__barplot.pdf".format(scenario, year, fueltype, region)
                    path_out_file = os.path.join(path_out_folder_fig4, fig_name)
                    seperate_legend = True
                    if seperate_legend:
                        export_legend(
                            legend,
                            os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                        legend.remove()

                    plt.tight_layout()

                    plt.savefig(path_out_file, transparent=True, bbox_inches='tight')
                    clear_figure(plt, fig, ax)

                # ----------------------------
                # Plot legend element
                # ----------------------------
                table_out = []
                fig, ax = plt.subplots()

                dummy_df = pd.DataFrame([[interval]], columns=['test'])

                ax = dummy_df.plot(
                    kind='bar',
                    y=dummy_df.columns,
                    color='black',
                    width=0.4)
    
                plt.yticks(
                    ticks=ticks,
                    labels=labels,
                    fontsize=8)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(False)

                #plt.xlabel("TW interval {}".format(interval))
                plt.tick_params(axis='x', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)

                # Rest size
                fig = plt.gcf()
                fig.set_size_inches(cm2inch(widht, height))

                ax.legend().set_visible(False)

                fig_name = "{}_{}_{}__barplot_dimension_legend.pdf".format(scenario, year, fueltype)
                path_out_file = os.path.join(path_out_folder_fig4, fig_name)

                plt.savefig(path_out_file, transparent=True, bbox_inches='tight')
                clear_figure(plt, fig, ax)

                table_tabulate = tabulate(
                    table_out,
                    headers=headers,
                    numalign="right")
                write_to_txt(path_out_file[:-4] + ".txt", table_tabulate)

            # ========================================================
            # PLot pie-charts
            # ========================================================
            radius_terawatt = 80 # 100% (radius 1) corresponds to 15 Terawatt (used to configure size of pie-charts)

            for scenario in scenarios:
                table_out = []
                for mode in [right, left]:

                    data_pie_chart = fig_dict_piecharts[year][mode][scenario]

                    # Temporal conversion
                    data_pie_chart = data_pie_chart * temporal_conversion_factor

                    #  Calculate new radius depending on demand (area proportional to size) (100%)
                    initial_radius = 1
                    total_sum = data_pie_chart.sum() / 1000 #Terawatt
                    area_change_p = total_sum / radius_terawatt

                    # Convert that radius reflects the change in area (and not size)
                    new_radius = math.sqrt(area_change_p) * initial_radius

                    # write results to txt
                    for index in data_pie_chart.index:
                        absolute = data_pie_chart.loc[index]
                        relative = (absolute / total_sum)
                        table_out.append([mode, index, absolute, relative])

                    # Explode distance
                    explode_factor = new_radius * 0.1
                    explode_distance = [explode_factor for i in range(len(data_pie_chart.index))]

                    # ========================================================
                    # Plotting pie chart
                    # ========================================================
                    fig, ax = plt.subplots(figsize=cm2inch(4.5, 5))

                    if not annote_crit:
                        data_pie_chart.plot(
                            kind='pie',
                            labels=None,
                            explode=explode_distance,
                            radius=new_radius,
                            wedgeprops=dict(width=new_radius * 0.4),
                            colors=colors_xy_plot)
                    else:
                        # ---------------------
                        # Plot annotations of pie chart
                        # ---------------------
                        min_label_crit = 1 #[%] Minimum label criterium 
                        labels_p = data_pie_chart.values / total_sum
                        labels = labels_p.round(3) * 100 #to percent

                        wedges, texts = plt.pie(
                            data_pie_chart.values,
                            explode=explode_distance,
                            radius=new_radius,
                            wedgeprops=dict(width=0.4),
                            textprops=dict(color="b"))

                        for i, p in enumerate(wedges):
                            ang = (p.theta2 - p.theta1)/2. + p.theta1
                            y = np.sin(np.deg2rad(ang))
                            x = np.cos(np.deg2rad(ang))
                            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
 
                            if labels[i] > min_label_crit: #larger than 1 percent
                                ax.annotate(
                                    labels[i],
                                    xy=(x, y),
                                    xytext=((new_radius + explode_distance[i] + 0.2)*np.sign(x), 1.4*y),
                                    horizontalalignment=horizontalalignment,
                                    color='grey',
                                    size=8,
                                    weight="bold")

                        '''value_crit = 5.0 # [%] Only labes larger than crit are plotted

                        wedges, texts = plt.pie(
                            data_pie_chart.values,
                            explode=explode_distance,
                            radius=new_radius,
                            wedgeprops=dict(width=0.5),
                            startangle=-40)

                        bbox_props = dict(boxstyle="square, pad=0.1", fc="w", ec="k", lw=0)
                        kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
                                bbox=bbox_props, zorder=0, va="center")

                        # Round
                        labels_p = data_pie_chart.values / total_sum
                        labels = labels_p.round(3)
    
                        for i, p in enumerate(wedges):
                            ang = (p.theta2 - p.theta1)/2. + p.theta1
                            y = np.sin(np.deg2rad(ang))
                            x = np.cos(np.deg2rad(ang))
                            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                            kw["arrowprops"].update({"connectionstyle": connectionstyle})

                            value = labels[i] * 100 #to percent

                            if value > value_crit: #larger than 1 percent
                                ax.annotate(value, xy=(x, y), xytext=((new_radius + 0.4)*np.sign(x), 1.4*y),
                                            horizontalalignment=horizontalalignment, **kw)
                        '''

                    plt.xlabel('')
                    plt.ylabel('')

                    # Legend
                    legend = plt.legend(
                        labels=data_pie_chart.index,
                        ncol=2,
                        prop={'size': 10},
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        frameon=False)

                    # Save pdf of figure and legend
                    fig_name = "{}_{}_{}_{}__pie.pdf".format(scenario, year, fueltype, mode)
                    path_out_file = os.path.join(path_out_folder_fig3, fig_name)

                    if seperate_legend:
                        export_legend(
                            legend,
                            os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                        legend.remove()

                    # Remove frame
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    plt.savefig(path_out_file)
                    clear_figure(plt, fig, ax)

                    table_tabulate = tabulate(
                        table_out, headers=['mode', 'type', 'absolute', 'relative'],
                        numalign="right")
                    write_to_txt(path_out_file[:-4] + ".txt", table_tabulate)
                   
                    # Write total sum to txt file
                    write_to_txt(path_out_file[:-4] + "total_sum.txt", str(total_sum))

            # ========================================================
            # Plot x-y graph
            # ========================================================
            for scenario in scenarios:
                print("... plotting xy-graph: {}   {}".format(year, scenario))
                table_out = []

                # Data and plot
                df_right = fig_dict[year][right][scenario]
                df_left = fig_dict[year][left][scenario]
                
                headers = list(df_right.columns)
                headers.insert(0, "hour")
                headers.insert(0, "type")
                for index_hour in df_right.index:
                    row = list(df_right.loc[index_hour])
                    row.insert(0, index_hour)
                    row.insert(0, 'right')
                    table_out.append(row)

                for index_hour in df_left.index:
                    row = list(df_left.loc[index_hour])
                    row.insert(0, index_hour)
                    row.insert(0, 'left')
                    table_out.append(row)

                # Switch axis
                df_left = df_left * -1 # Convert to minus values
                #df_to_plot.plot.area()
                #df_to_plot.plot.bar(stacked=True)#, orientation='vertical')
                table_out.append([])

                fig, ax = plt.subplots(figsize=cm2inch(9, height_cm_xy_figure))

                # Bin infos
                nr_of_bins = x_values_lims[fueltype]['nr_of_bins']
                bin_value = x_values_lims[fueltype]['bin_value']

                # --------------
                # Sorting (make that tran_E is first entry)
                # --------------
                first_element_to_plot = 'eh_tran_e_export'
                attribute_to_plot = 'eh_tran_e_export' #Line plot argument
                color_to_plot_attribute = 'magenta' #Color of line plot argument

                try:
                    
                    _ = df_right[first_element_to_plot]
                    orig_order = df_right.columns.values.tolist()
                    orig_order.remove(first_element_to_plot)
                    orig_order.insert(0, first_element_to_plot)

                    # Reorder
                    df_right = df_right[orig_order]
                    df_left = df_left[orig_order]

                    # Remove attribute '' from dataframe
                    df_line_attribute_right = df_right[first_element_to_plot]
                    df_line_attribute_left = df_left[first_element_to_plot]

                    # Plot attribute '' as stepped line chart right

                    x_right = df_line_attribute_right.values.tolist()  
                    
                    y_right = [(bin_width * i) for i in range(nr_of_bins + 1)]
                    y_right = [i - 0.5 for i in range(len(hours_selected))]

                    x_left = df_line_attribute_left.values.tolist()  
                    y_left = [i - 0.5 for i in range(len(hours_selected))]

                    ax.step(x=x_right, y=y_right, zorder=4, linestyle='--', color=color_to_plot_attribute)
                    ax.step(x=x_left, y=y_left, zorder=4, linestyle='--', color=color_to_plot_attribute)
                    
                    # Remove from dataframe
                    df_right.drop([attribute_to_plot], axis=1)
                    df_left.drop([attribute_to_plot], axis=1)
                    #plt.show()
                except:
                    print("____dd_____")
                    pass #no first_element_to_plot in data

                df_right.plot(
                    kind='barh',
                    ax=ax,
                    width=1.0,
                    legend=True,
                    stacked=True,
                    color=colors_xy_plot,
                    zorder=1)

                df_left.plot(
                    kind='barh',
                    ax=ax,
                    width=1.0,
                    legend=False,
                    stacked=True, 
                    color=colors_xy_plot,
                    zorder=1)

                # Add vertical line
                ax.axvline(linewidth=1, color='black', zorder=3)

                # Customize x-axis
                right_ticks = np.array([bin_value * i for i in range(nr_of_bins + 2)])
                left_ticks = right_ticks * -1
                left_ticks = left_ticks[::-1]
                right_labels = [str(bin_value * i) for i in range(nr_of_bins + 2)]
                left_labels = right_labels[::-1]
                ticks = list(left_ticks) + list(right_ticks)
                labels = list(left_labels) + list(right_labels)

                plt.xticks(
                    ticks=ticks,
                    labels=labels,
                    fontsize=fontsize_small)

                # Customize y-axis
                nr_of_bins = 4
                bin_width = int(len(hours_selected) / nr_of_bins)
                min_bin_value = int(np.min(hours_selected))

                ticks = np.array([(bin_width * i) for i in range(nr_of_bins + 1)])
                labels = np.array([str(min_bin_value + bin_width * i) for i in range(nr_of_bins + 1)])

                plt.yticks(
                    ticks=ticks,
                    labels=labels,
                    fontsize=fontsize_small)

                # Legend
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles)) # remove duplicates
                legend = plt.legend(
                    by_label.values(),
                    by_label.keys(),
                    ncol=2,
                    prop={'size': 10},
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.1),
                    frameon=False)

                # Remove frame
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                # Save pdf of figure and legend
                fig_name = "{}_{}_{}__xy_plot.pdf".format(scenario, year, fueltype)
                path_out_file = os.path.join(path_out_folder_fig3, fig_name)

                if seperate_legend:
                    export_legend(
                        legend,
                        os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                    legend.remove()

                # Add grid lines
                ax.grid(which='major', color='white', axis='x', linestyle='--')
                plt.tick_params(axis='x', which='both', bottom=False) #remove ticks
                plt.tick_params(axis='y', which='both', left=False) #remove ticks

                # Labels
                # ------------
                plt.xlabel("{}".format(unit), fontdict=font_additional_info)
                #plt.ylabel("Time: {}".format(seasonal_week_day),  fontdict=font_additional_info)
                plt.ylabel("Hour of peak day")
                plt.savefig(path_out_file)
                clear_figure(plt, fig, ax)

                table_tabulate = tabulate(
                    table_out,
                    headers=headers,
                    numalign="right")
                write_to_txt(path_out_file[:-4] + ".txt", table_tabulate)


def fig_4(data_container):

    return


def fig_5(
        path_out,
        data_container,
        filenames,
        scenarios,
        weather_scearnio,
        types_to_plot,
        unit,
        years=[2015, 2030, 2050],
        seperate_legend=True
    ):
    """
    """

    return


def fig_6(data_container):
    """
    """

    return
