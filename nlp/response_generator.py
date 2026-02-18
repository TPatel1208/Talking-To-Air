# response_generator.py

from typing import Dict, Any, List, Optional


class ResponseGenerator:
    """Generate natural language responses"""
    
    def __init__(self):
        # Pollutant display names
        self.pollutant_labels = {
            'no2': 'Nitrogen Dioxide (NO₂)',
            'pm2.5': 'Fine Particulate Matter (PM2.5)',
            'pm10': 'Particulate Matter (PM10)',
            'o3': 'Ozone (O₃)',
            'so2': 'Sulfur Dioxide (SO₂)',
            'co': 'Carbon Monoxide (CO)',
        }
        
        # Statistic display names
        self.stat_labels = {
            'mean': 'Average',
            'max': 'Maximum',
            'min': 'Minimum',
            'median': 'Median',
        }
        
        # Clarification templates
        self.clarification_templates = {
            'pollutant': (
                "Which pollutant would you like to check?\n"
                "Options: NO2, PM2.5, PM10, O3, SO2, CO"
            ),
            'location': (
                "Which location should I analyze?\n"
                "(You can specify a city name, coordinates, or region)"
            ),
            'time': (
                "For which time period?\n"
                "(Examples: 'today', 'yesterday', 'last week', 'July 2024')"
            ),
            'statistic': (
                "How should I aggregate the data?\n"
                "Options: mean, max, min, median"
            ),
        }
        
        # Intent-specific confirmation templates
        self.confirmation_templates = {
            'STATISTIC': {
                'standard': "Analyzing {stat} {pollutant} levels in {location} for {time}...",
                'with_context': "Continuing with {location}. Checking {stat} {pollutant} for {time}...",
                'assumed_values': "Checking {stat} {pollutant} in {location} for {time} (based on your recent queries)..."
            },
            'MAP_PLOT': {
                'standard': "Generating {pollutant} map for {location} in {time}...",
                'with_context': "Continuing with {location}. Creating {pollutant} map for {time}...",
                'assumed_values': "Creating {pollutant} map for {location} in {time} (based on your recent queries)..."
            },
            'TIME_SERIES': {
                'standard': "Analyzing {pollutant} trends in {location} for {time}...",
                'with_context': "Continuing with {location}. Analyzing {pollutant} trends for {time}...",
                'assumed_values': "Analyzing {pollutant} trends in {location} for {time} (based on your recent queries)..."
            },
            'TEMPORAL_ANALYSIS': {
                'standard': "Analyzing {pollutant} temporal patterns in {location} for {time}...",
                'with_context': "Continuing with {location}. Analyzing {pollutant} temporal patterns for {time}...",
                'assumed_values': "Analyzing {pollutant} temporal patterns in {location} for {time} (based on your recent queries)..."
            },
            'QUERY': {
                'standard': "Looking up {pollutant} data for {location} in {time}...",
                'with_context': "Continuing with {location}. Fetching {pollutant} data for {time}...",
                'assumed_values': "Fetching {pollutant} data for {location} in {time} (based on your recent queries)..."
            },
            'COMPARISON': {
                'standard': "Comparing {pollutant} levels across locations for {time}...",
                'with_context': "Continuing comparison. Checking {pollutant} for {time}...",
                'assumed_values': "Comparing {pollutant} in {time} (based on your recent queries)..."
            },
        }
    
    def generate_clarification(
        self, 
        missing_slots: List[str],
        current_params: Dict[str, Any]
    ) -> str:
        """
        Generate a clarification question for missing information.
        
        Args:
            missing_slots: List of slot names that are missing (e.g., ['location', 'time'])
            current_params: What we already know (e.g., {'pollutant_type': 'no2'})
            
        Returns:
            Natural language question asking for the first missing slot
            
        Example:
            >>> rg.generate_clarification(['location'], {'pollutant_type': 'no2'})
            "Which location should I analyze?
             (You can specify a city name, coordinates, or region)
             
             (I have: pollutant: NO2)"
        """
        
        if not missing_slots:
            return "I have all the information I need!"
        
        # Get the first missing slot
        slot = missing_slots[0]
        
        # Get the template question for this slot
        question = self.clarification_templates.get(
            slot, 
            f"Could you specify the {slot}?"
        )
        
        # Build list of what we already know
        known_parts = []
        
        if current_params.get('pollutant_type'):
            known_parts.append(f"pollutant: {current_params['pollutant_type'].upper()}")
        
        if current_params.get('location'):
            known_parts.append(f"location: {current_params['location']}")
        
        if current_params.get('time_filter'):
            time_str = self._format_time(current_params['time_filter'])
            known_parts.append(f"time: {time_str}")
        
        if current_params.get('aggregation'):
            known_parts.append(f"aggregation: {current_params['aggregation']}")
        
        # Add what we know to the question
        if known_parts:
            question += f"\n\n(I have: {', '.join(known_parts)})"
        
        return question
    
    def generate_confirmation(
        self,
        intent: str,
        query_params: Dict[str, Any],
        context_used: Dict[str, str]
    ) -> str:
        """
        Generate confirmation message based on intent type.
        
        Args:
            intent: The classified intent (STATISTIC, MAP_PLOT, etc.)
            query_params: Query parameters
            context_used: Which params came from context (e.g., {'location': 'previous_query'})
            
        Returns:
            Confirmation message
        """
        
        # Get templates for this intent (fallback to QUERY if unknown)
        templates = self.confirmation_templates.get(
            intent, 
            self.confirmation_templates['QUERY']
        )
        
        # Choose template based on context usage
        if not context_used:
            template = templates['standard']
        elif len(context_used) == 1:
            template = templates['with_context']
        else:
            template = templates['assumed_values']
        
        # Format parameters
        formatted = self._format_params(query_params)
        
        try:
            return template.format(**formatted)
        except KeyError as e:
            # Fallback if template has missing keys
            return f"Processing {formatted.get('pollutant', 'data')} query for {formatted.get('location', 'location')}..."
    
    def generate_result(
        self,
        intent: str,
        query_params: Dict[str, Any],
        result: Dict[str, Any],
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate final result response.
        
        Args:
            intent: The classified intent
            query_params: Query parameters
            result: Query result with 'value' and 'unit'
            history: Previous queries for comparison (optional)
            
        Returns:
            Formatted result string
        """
        
        # Handle errors
        if 'error' in result:
            formatted = self._format_params(query_params)
            return (
                f"Sorry, no data available for {formatted['pollutant']} "
                f"in {formatted['location']} during {formatted['time']}."
            )
        
        # Format parameters
        formatted = self._format_params(query_params)
        
        # Build response
        value = result.get('value')
        unit = result.get('unit', 'µg/m³')
        
        if value is None:
            response = f"No data available for {formatted['pollutant_full']} in {formatted['location']} during {formatted['time']}."
        else:
            response = (
                f"{formatted['pollutant_full']} in {formatted['location']} "
                f"during {formatted['time']}: {value:.2f} {unit}"
            )
            
            # Add health assessment
            health = self._assess_health(
                query_params.get('pollutant_type'),
                value
            )
            response += f" — {health}"
            
            # Add comparison if available
            if history:
                comparison = self._generate_comparison(
                    query_params,
                    result,
                    history
                )
                if comparison:
                    response += f". {comparison}"
        
        return response
    
    def generate_low_confidence(self, confidence: float) -> str:
        """
        Generate response for low confidence intent classification.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Message asking user to rephrase
        """
        return (
            f"I'm not quite sure I understood that (confidence: {confidence:.0%}).\n"
            "Could you rephrase?\n"
            "Example: 'What was the average NO2 in Paris last month?'"
        )
    
    def _format_params(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Format parameters for display"""
        
        pollutant = params.get('pollutant_type', 'pollutant')
        
        return {
            'pollutant': pollutant.upper(),
            'pollutant_full': self.pollutant_labels.get(
                pollutant.lower(), 
                pollutant.upper()
            ),
            'location': params.get('location', 'the area'),
            'time': self._format_time(params.get('time_filter', {})),
            'stat': params.get('aggregation', 'mean'),
            'stat_label': self.stat_labels.get(
                params.get('aggregation', 'mean'), 
                'Average'
            ),
        }
    
    def _format_time(self, time_filter) -> str:
        """Format time filter to readable string"""
        if time_filter is None:
            return 'the period'
        
        if isinstance(time_filter, dict):
            parsed = time_filter.get('parsed', time_filter.get('raw_text', 'the period'))
            return str(parsed)
        
        return str(time_filter)
    
    def _assess_health(self, pollutant: str, value: float) -> str:
        """Assess health based on WHO guidelines"""
        
        if not pollutant or value is None:
            return "data available"
        
        # WHO thresholds (µg/m³)
        thresholds = {
            'no2': {'good': 25, 'moderate': 50, 'unhealthy': 100},
            'pm2.5': {'good': 15, 'moderate': 35, 'unhealthy': 55},
            'pm10': {'good': 45, 'moderate': 75, 'unhealthy': 150},
            'o3': {'good': 100, 'moderate': 160, 'unhealthy': 200},
            'so2': {'good': 40, 'moderate': 100, 'unhealthy': 350},
            'co': {'good': 4000, 'moderate': 9000, 'unhealthy': 15000},
        }
        
        thresh = thresholds.get(pollutant.lower(), {})
        
        if value < thresh.get('good', float('inf')):
            return "Good air quality"
        elif value < thresh.get('moderate', float('inf')):
            return "Moderate air quality"
        elif value < thresh.get('unhealthy', float('inf')):
            return "Unhealthy for sensitive groups"
        else:
            return "Unhealthy air quality"
    
    def _generate_comparison(
        self,
        current_params: Dict[str, Any],
        current_result: Dict[str, Any],
        history: List[Dict]
    ) -> Optional[str]:
        """Generate comparison with previous query"""
        
        if not history:
            return None
        
        current_pollutant = current_params.get('pollutant_type')
        current_location = current_params.get('location')
        current_value = current_result.get('value')
        
        if not current_value:
            return None
        
        # Find comparable query (same pollutant and location)
        for past in reversed(history):
            past_params = past.get('params', {})
            
            if (past_params.get('pollutant_type') == current_pollutant and
                past_params.get('location') == current_location):
                
                past_result = past.get('result', {})
                past_value = past_result.get('value')
                
                if past_value:
                    # Calculate change
                    change_pct = ((current_value - past_value) / past_value) * 100
                    
                    if abs(change_pct) < 5:
                        change_text = "similar"
                    elif change_pct > 0:
                        change_text = f"{change_pct:.1f}% higher"
                    else:
                        change_text = f"{abs(change_pct):.1f}% lower"
                    
                    past_time = self._format_time(past_params.get('time_filter', {}))
                    
                    return f"This is {change_text} compared to {past_time}"
        
        return None