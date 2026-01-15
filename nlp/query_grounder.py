from typing import Dict, Any
from nlp.slot_extractor import SlotExtractor

class QueryGrounder:    
    def __init__(self, slot_extractor: SlotExtractor):
        self.slot_extractor = slot_extractor
    
    def ground_query(self, query: str, intent: str) -> Dict[str, Any]:
        """Ground a natural language query to structured parameters"""
        slots = self.slot_extractor.extract_slots(query, intent)
        
        grounded = {
            'intent': intent,
            'slots': slots,
            'query_params': self._build_query_params(slots),
            'validation': self._validate_grounding(slots, intent)
        }
        
        return grounded
    
    def _build_query_params(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        
        if slots['pollutant']:
            params['pollutant_type'] = slots['pollutant']['type']
        
        if slots['location']:
            params['location'] = slots['location']['name']
            params['location_type'] = slots['location']['type']
        
        if slots['statistic']:
            params['aggregation'] = slots['statistic']['type']
        
        if slots['time']:
            params['time_filter'] = slots['time']
            params['temporal_type'] = slots['time']['type']
        
        if slots['comparison']:
            params['comparison_mode'] = True
        
        return params
    
    def _validate_grounding(self, slots: Dict[str, Any], intent: str) -> Dict[str, Any]:
        validation = {
            'is_valid': True,
            'missing_slots': [],
            'warnings': []
        }
        
        # Define required slots per intent
        required_slots = {
            'MAP_PLOT': ['pollutant', 'location'],
            'STATISTIC': ['pollutant', 'location', 'statistic'],
            'TEMPORAL_ANALYSIS': ['pollutant','location', 'statistic'],
        }
        
        if intent in required_slots:
            for required in required_slots[intent]:
                if not slots.get(required):
                    validation['is_valid'] = False
                    validation['missing_slots'].append(required)
        
        #can be used for future llm 
        if slots['confidence'] < 0.6:
            validation['warnings'].append('Low overall confidence in slot extraction')
        
        return validation


    