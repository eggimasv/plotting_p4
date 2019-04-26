"""Functions to generate figures
"""
import os
import math
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
from tabulate import tabulate

# Set default font
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
#rcParams.update({'figure.autolayout': True})

def autolabel(rects, ax, rounding_digits, value_show_crit=0.01):
    """Place labels with percentages in middle of stacked plots
    """
    for rect in rects:
        value = rect.get_width()
        value_round = round(value, rounding_digits) * 100

        if value_round > value_show_crit:
            ax.text(rect.get_x() + rect.get_width() / 2,#width
                    rect.get_y() + rect.get_height() / 2.4, #height
                    s=value_round,
                    ha='center',
                    va='center',
                    color="White",
                    fontsize=8,
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

def load_data(in_path, scenarios, simulation_name, unit):
    """Read results and set timestep as index in dataframe

    Returns
    data_container : dataframes in [scenario][mode][weather_scenario]
    """
    data_container = {}
    modes = ['CENTRAL', 'DECENTRAL']

    for scenario in scenarios:
        data_container[scenario] = {}

        for mode in modes:
            data_container[scenario][mode] = {}

            # Iterate over weather scenarios
            path_scenarios = os.path.join(in_path, scenario, mode)
            weather_scenarios = os.listdir(path_scenarios)

            for weather_scenario in weather_scenarios:
                data_container[scenario][mode][weather_scenario] = {}

                # ---------------------------
                # Load supply generation mix
                # ---------------------------
                data_container[scenario][mode][weather_scenario] ['energy_supply_constrained'] = {}
                path_supply_mix = os.path.join(in_path, scenario, mode, weather_scenario, simulation_name, 'energy_supply_constrained', 'decision_0')

                all_files = os.listdir(path_supply_mix)

                for file_name in all_files:
                    path_file = os.path.join(path_supply_mix, file_name)
                    print(".... loading file: {}".format(file_name))
                    variable_name = file_name[7:-18]
                
                    # Load data
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

    return data_container

def plot_figures(
        path_out,
        data_container,
        filenames,
        scenarios,
        weather_scearnio,
        types_to_plot,
        unit,
        temporal_conversion_factor,
        years=[2015, 2030, 2050],
        seperate_legend=True
    ):
    """Create x-y chart of a time-span (x-axis: demand, y-axis: time)
    """
    for fueltype in types_to_plot:
        print(".... fueltype: {}".format(fueltype))

        annote_crit = False #Add labels

        # Select hours to plots
        height_cm_xy_figure = 5 # Figure height of xy figure

        seasonal_week_day = 2
        hours_selected = range(24 * (seasonal_week_day) + 1, 24 * (seasonal_week_day + 1) + 1)

        # Select a full week
        #height_cm_xy_figure = 10 # Figure height of xy figure
        #hours_selected = range(24 * (0) + 1, 24 * (6 + 1) + 1)

        modes = ['DECENTRAL', 'CENTRAL']
        left = 'CENTRAL'
        right = 'DECENTRAL'

        fontsize_small = 8
        fontsize_large = 10

        # Font info axis labels
        font_additional_info = {
            'color': 'black',
            'weight': 'bold',
            'size': fontsize_large}

        #https://www.color-hex.com/color-palette/77223
        fueltypes_coloring = {
            'electricity': '#03674f',
            'gas': '#669be6',
            'heat': '#e9c0fd',
            'hydrogen': '#00242b'}

        fig_dict = {}
        fig_dict_piecharts = {}
        fig_dict_fuelypes = {}
        fig_dict_regional_annual_demand = {}

        path_out_folder_fig3 = os.path.join(path_out, 'fig3')
        path_out_folder_fig4 = os.path.join(path_out, 'fig4')

        for year in years:
            fig_dict[year] = {}
            fig_dict_piecharts[year] = {}
            fig_dict_fuelypes[year] = {}
            fig_dict_regional_annual_demand[year] = {}

            for mode in modes:
                fig_dict[year][mode] = {}
                fig_dict_piecharts[year][mode] = {}
                fig_dict_fuelypes[year][mode] = {}
                fig_dict_regional_annual_demand[year][mode] = {}

                for scenario in scenarios:
                    fig_dict[year][mode][scenario] = {}
                    fig_dict_fuelypes[year][mode][scenario] = pd.DataFrame(
                        [[0,0,0,0]], columns=fueltypes_coloring.keys())

                    colors = []
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
                                colors.append(color)
                            
                            # Get fueltype specific files
                            sum_file = national_per_timesteps[name_column].sum()

                            if (file_name_split_no_timpestep in filenames['elec_hubs'].keys()) or (
                                file_name_split_no_timpestep in filenames['elec_transmission'].keys()):
                                fig_dict_fuelypes[year][mode][scenario]['electricity'] += sum_file
                            elif (file_name_split_no_timpestep in filenames['gas_hubs'].keys()) or (
                                file_name_split_no_timpestep in filenames['gas_transmission'].keys()):
                                fig_dict_fuelypes[year][mode][scenario]['gas'] += sum_file
                            elif file_name_split_no_timpestep in filenames['heat_hubs'].keys():
                                fig_dict_fuelypes[year][mode][scenario]['heat'] += sum_file
                            elif file_name_split_no_timpestep in filenames['hydrogen_hubs'].keys():
                                fig_dict_fuelypes[year][mode][scenario]['hydrogen'] += sum_file

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
    
                colors_right_left = {
                    right: '#ddca7c',
                    left: '#4a8487'}

                ax = df_bars.plot(
                    kind='bar',
                    width=0.8,
                    color=list(colors_right_left.values()))

                # Remove frame
                # ------------
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(False)

                #Axis label
                ax.set_xlabel('Energy Hub Region')
                ax.set_ylabel('GW')

                fig_name = "{}_{}_{}__barplots_comparison_all.pdf".format(scenario, year, fueltype)
                path_out_file = os.path.join(path_out_folder_fig4, fig_name)
                plt.savefig(path_out_file, transparent=True, bbox_inches='tight')

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
                        x=df_bars.values,
                        y=df_bars.columns,
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
                    ax.grid(which='major', color='white', axis='y', linestyle='-')
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
                    height=2
                    fig = matplotlib.pyplot.gcf()
                    fig.set_size_inches(cm2inch(widht, height))

                    #plt.autoscale(enable=True, axis='x', tight=True)
                    #plt.autoscale(enable=True, axis='y', tight=True)

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
                    #plt.show()

                    plt.savefig(path_out_file, transparent=True, bbox_inches='tight')

                # ----------------------------
                # Plot Legend element
                # ----------------------------
                table_out = []
                fig, ax = plt.subplots()

                dummy_df = pd.DataFrame([[interval]], columns=['test'])

                ax = dummy_df.plot(
                    kind='bar',
                    x=dummy_df.values,
                    y=dummy_df.columns,
                    width=0.4)
    
                plt.yticks(
                    ticks=ticks,
                    labels=labels,
                    fontsize=8)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(False)

                plt.xlabel("TW interval {}".format(interval))

                # Rest size
                fig = matplotlib.pyplot.gcf()
                fig.set_size_inches(cm2inch(widht, height))

                ax.legend().set_visible(False)
    
                fig_name = "{}_{}_{}__barplot_dimension_legend.pdf".format(scenario, year, fueltype)
                path_out_file = os.path.join(path_out_folder_fig4, fig_name)
 
                plt.savefig(path_out_file, transparent=True, bbox_inches='tight')

                # Write out results to txt
                table_tabulate = tabulate(
                    table_out,
                    headers=headers,
                    numalign="right")
                write_to_txt(path_out_file[:-4] + ".txt", table_tabulate)

            # ----------------------
            # Fueltype chart showing the split between fueltypes
            # ----------------------
            for scenario in scenarios:
                table_out = []
                for mode in [right, left]:

                    fig, ax = plt.subplots()

                    data_fueltypes = fig_dict_fuelypes[year][mode][scenario]

                    # Conver to percentages
                    data_fueltypes_p = data_fueltypes / data_fueltypes.sum().sum()
                    data_fueltypes_p = data_fueltypes_p.round(2)

                    absolute_values = list(data_fueltypes.values[0].tolist())
                    relative_values = data_fueltypes_p.values[0].tolist()

                    absolute_values.insert(0, 'absolute')
                    relative_values.insert(0, 'relative')
                    absolute_values.insert(0, mode)
                    relative_values.insert(0, mode)
                    table_out.append(absolute_values)
                    table_out.append(relative_values)

                    headers = list(data_fueltypes_p.columns)
                    headers.insert(0, 'type')
                    headers.insert(0, 'mode')

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
                    plt.yticks(
                        ticks=[0],
                        labels=[''],
                        fontsize=fontsize_small)

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
                    fig = matplotlib.pyplot.gcf()
                    fig.set_size_inches(cm2inch(4.0, 0.3))

                    # Remove frame
                    # ------------
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    plt.savefig(path_out_file)

                    # Write out results to txt
                    table_tabulate = tabulate(
                        table_out,
                        headers=headers,
                        numalign="right")
                    write_to_txt(path_out_file[:-4] + ".txt", table_tabulate)

            # ----------------------
            # PLot pie-charts
            # ----------------------
            radius_terawatt = 80 # 100% (radius 1) corresponds to 15 Terawatt (used to configure size of pie-charts)

            for scenario in scenarios:
                table_out = []
                for mode in [right, left]:

                    data_pie_chart = fig_dict_piecharts[year][mode][scenario]

                    # Temporal conversion
                    data_pie_chart = data_pie_chart * temporal_conversion_factor

                    #  Calculate new radius depending on demand (area proportional to size) (100%)
                    initial_radius = 1
                    total_sum = data_pie_chart.sum() / 1000
                    area_change_p = total_sum / radius_terawatt

                    # Convert that radius reflects the change in area (and not size)
                    new_radius = math.sqrt(area_change_p) * initial_radius

                    # write results to txt
                    total_sum = data_pie_chart.sum()
                    for index in data_pie_chart.index:
                        absolute = data_pie_chart.loc[index]
                        relative = (absolute / total_sum)
                        table_out.append([mode, index, absolute, relative])

                    # Explode distance
                    explode_factor = new_radius * 0.1
                    explode_distance = [explode_factor for i in range(len(data_pie_chart.index))]

                    # ---------------------
                    # Plotting pie chart
                    # ---------------------
                    fig, ax = plt.subplots(figsize=cm2inch(4.5, 5))

                    if not annote_crit:
                        plt.pie(
                            data_pie_chart.values,
                            explode=explode_distance,
                            radius=new_radius,
                            wedgeprops=dict(width=new_radius * 0.4))
                    else:
                        # ---------------------
                        # Plot annotations of pie chart
                        # ---------------------
                        min_label_crit = 1 #[%] Minimum label criterium 
                        # Round
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

                    # Labels
                    plt.xlabel('')
                    plt.ylabel('')

                    # Legend
                    # ------------
                    legend = plt.legend(
                        labels=data_pie_chart.index,
                        ncol=2,
                        prop={'size': 10},
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.1),
                        frameon=False)

                    # Save pdf of figure and legend
                    # ------------
                    fig_name = "{}_{}_{}_{}__pie.pdf".format(scenario, year, fueltype, mode)
                    path_out_file = os.path.join(path_out_folder_fig3, fig_name)

                    if seperate_legend:
                        export_legend(
                            legend,
                            os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                        legend.remove()

                    # Limits
                    # ------------
                    #plt.autoscale(enable=True, axis='x', tight=True)
                    #plt.autoscale(enable=True, axis='y', tight=True)
                    #plt.tight_layout()

                    # Remove frame
                    # ------------
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    #plt.show()
                    plt.savefig(path_out_file)

                    # Write out results to txt
                    table_tabulate = tabulate(
                        table_out, headers=['mode', 'type', 'absolute', 'relative'],
                        numalign="right")
                    write_to_txt(path_out_file[:-4] + ".txt", table_tabulate)

            # ----------
            # Plot x-y graph
            # ----------
            for scenario in scenarios:
                print("xy-graph: {}   {}".format(year, scenario))
                table_out = []

                # Data and plot
                # ------------
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
                df_right.plot(kind='barh', ax=ax, width=1.0, stacked=True, color=colors)
                df_left.plot(kind='barh', ax=ax, width=1.0, legend=False, stacked=True,  color=colors)

                # Add vertical line
                # ------------
                ax.axvline(linewidth=1, color='black')
                
                # Title
                # ------
                #plt.title(left, fontdict=None, loc='left', fontsize=fontsize_small)
                #plt.title(right, fontdict=None, loc='right', fontsize=fontsize_small)

                # Customize x-axis
                nr_of_bins = 4
                bin_value = int(np.max(df_right.values) / nr_of_bins)
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

                ticks = np.array([(bin_width * i) - 0.5 for i in range(nr_of_bins + 1)])
                labels = np.array([str(min_bin_value + bin_width * i) for i in range(nr_of_bins + 1)])

                plt.yticks(
                    ticks=ticks,
                    labels=labels,
                    fontsize=fontsize_small)

                # Legend
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

                # Remove frame
                # ------------
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                # Limits
                # ------------
                plt.autoscale(enable=True, axis='x', tight=True)
                plt.autoscale(enable=True, axis='y', tight=True)

                # Add grid lines
                # ------------
                ax.grid(which='major', color='black', axis='y', linestyle='--')
                plt.tick_params(axis='y', which='both', left=False) #remove ticks

                # Save pdf of figure and legend
                # ------------
                fig_name = "{}_{}_{}__xy_plot.pdf".format(scenario, year, fueltype)
                path_out_file = os.path.join(path_out_folder_fig3, fig_name)

                if seperate_legend:
                    export_legend(
                        legend,
                        os.path.join("{}__legend.pdf".format(path_out_file[:-4])))
                    legend.remove()

                # Labels
                # ------------
                plt.xlabel("{}".format(unit), fontdict=font_additional_info)
                #plt.ylabel("Time: {}".format(seasonal_week_day),  fontdict=font_additional_info)
                plt.ylabel("Hour of peak day")
                plt.savefig(path_out_file)
                
                # Write out results to txt
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
