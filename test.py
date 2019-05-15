import numpy as np
from datetime import date
from datetime import timedelta

def date_to_yearday(year, month, day):
    """Gets the yearday (julian year day) of a year minus one to correct because of python iteration

    Arguments
    ----------
    date_base_yr : int
        Year
    date_base_yr : int
        Month
    day : int
        Day

    Example
    -------
    5. January 2015 --> Day nr 5 in year --> -1 because of python --> Out: 4
    """
    date_y = date(year, month, day)
    yearday = date_y.timetuple().tm_yday - 1 #: correct because of python iterations

    return yearday

def get_seasonal_weeks():
    """
    """
    winter_2018 = list(range(date_to_yearday(2010, 1, 1), date_to_yearday(2019, 3, 19)))
    winter_2019 = list(range(date_to_yearday(2018, 12, 21), date_to_yearday(2018, 12, 31)))
    winter_2018.append(winter_2019)
    winter = winter_2018
        #range(date_to_yearday(2018, 12, 21), date_to_yearday(2019, 3, 19))) #Jan
    spring = list(
        range(
            date_to_yearday(2019, 3, 20),
            date_to_yearday(2019, 6, 20)))
    summer = list(
        range(
            date_to_yearday(2019, 6, 21),
            date_to_yearday(2019, 9, 22)))
    autumn = list(
        range(
            date_to_yearday(2019, 9, 23),
            date_to_yearday(2019, 12, 21)))

    return winter, spring, summer, autumn

'''
class ConvertHourlyToSeasonalWeek(SectorModel):
    """Parameterised interval sampling
    """
    def before_model_run(self, data_handle):
        """Set up coefficients/read from cache?
        """
    def simulate(self, data_handle):
        """Convert timesteps, using parameterised approach
        """        
        conversion_mode = data_handle.get_parameter('conversion_mode')
        for from_spec in self.inputs.values():
            if from_spec.name in self.outputs:
                to_spec = self.outputs[from_spec.name]
                data_in = data_handle.get_data(from_spec.name)
                data_out = self.convert(data_in, to_spec, conversion_mode)
                data_handle.set_results(to_spec.name, data_out)
    def convert(self, data, to_spec, conversion_mode):
        """Do conversion
        """
        ...
'''
reg_nrs = 3
empty_input_data = np.array((reg_nrs, 8760)) + 1

winter, spring, summer, autumn = get_seasonal_weeks()
print(len(winter))
print(len(spring))
print(len(summer))
print(len(autumn))

def convert_seasonal_week_actual_days():

    return

