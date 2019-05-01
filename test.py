import numpy as np

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
        ...'''
reg_nrs = 3
empty_input_data = np.array(reg_nrs, 8760) + 1

def convert_seasonal_week_actual_days():

    

