from typing import Dict, Any, List, Optional
from utils.statistics import compute_statistic, daily_cycle_peak, get_time_index_local
from utils.plotting import mask_data_by_geometry
import xarray as xr

"""
This currently does not handle time queries, due to the fact that I still have to get the data. However a database can be created later to handle time queries.
Currently using a default dataset for testing. Also multiplotting functionality is not implemented yet and will be added later.
"""
class QueryExecutor:
    def __init__ (self, region_resolver, default_ds: Optional[xr.Dataset] = None):
        self.region_resolver = region_resolver
        self.default_ds = default_ds
    def execute_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the query based on the intent and parameters."""
        intent = data.get('intent')
        query_params = data.get('query_params', {})
        if intent == 'MAP_PLOT':
            return self._execute_plotting_query(query_params)
        elif intent == 'TEMPORAL_ANALYSIS':
            return self._execute_temporal_query(query_params)
        elif intent == 'STATISTIC':
            return self._execute_statistical_query(query_params)
        else:
            return None

    def _execute_plotting_query(self, query_params: Dict[str, Any]) -> Any:
        # Placeholder for executing plotting queries
        time_filter = query_params.get('time_filter', None)
        pollutant = query_params.get('pollutant_type', 'NO2')
        location = query_params.get('location', None)

        print(f"Executing plotting query for {pollutant} at {location} with time filter {time_filter['parsed'] if time_filter else 'None'}")
        fig, ax = self.region_resolver.plot_singular(
            self.default_ds, 
            location,
            title=f"{pollutant} over {location}",
            time_slice=6
        )
        # Implementation would go here
        return {'PLOT':(fig, ax)}

    def _execute_temporal_query(self, query_params: Dict[str, Any]) -> Any:
        """Currently only supports temporal max"""
        time_filter = query_params.get('time_filter', None)
        pollutant = query_params.get('pollutant_type', 'NO2')
        location = query_params.get('location', None)
        temporal_type = query_params.get('temporal_type', 'month')
        aggregation = query_params.get('aggregation', 'mean')
        print(f"Executing temporal query: over {temporal_type} with {aggregation} for {pollutant} at {location} with time filter {time_filter['parsed'] if time_filter else 'None'}")
        # Implementation would go here
        result = self.region_resolver.resolve_location(location)
        masked_data = mask_data_by_geometry(self.default_ds, result['geometry'])
        peak_hour, peak_val = daily_cycle_peak(masked_data)
        return {'TEMPORAL': f"Peak at {peak_hour}:00 with value {peak_val:.2f}"}
    

    def _execute_statistical_query(self, query_params: Dict[str, Any]) -> Any:
        # Placeholder for executing statistical queries
        time_filter = query_params.get('time_filter', None)
        pollutant = query_params.get('pollutant_type', 'NO2')
        location = query_params.get('location', None)
        aggregation = query_params.get('aggregation', 'mean')

        print(f"Executing statistical query for {pollutant} at {location} with aggregation {aggregation} and time filter {time_filter['parsed'] if time_filter else 'None'}")
        result = self.region_resolver.resolve_location(location)
        masked_data = mask_data_by_geometry(self.default_ds, result['geometry'])
        statistic_value = compute_statistic(
            data_array=masked_data,
            statistic=aggregation,
        )
        return {'STATISTIC': f"{aggregation.capitalize()} {pollutant} level in {location} is {statistic_value}"}