from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

class Dialouge_Manager:
    def __init__(self, intent_embedder, slot_extractor, query_grounder):
        self.intent_embedder = intent_embedder
        self.slot_extractor = slot_extractor
        self.query_grounder = query_grounder
        self.sessions = {} # session_id -> context_dict

    def process_query(self, session_id: str, query: str) -> Tuple[str, Dict[str, Any]]:

        
        intent, score, matches = self.intent_embedder.predict(query)
        grounded = self.query_grounder.ground_query(query, intent)
        ctx = self._get_session_context(session_id)

        
        #store in conversation history
        ctx['messages'].append({
            'user': query,
            'timestamp': datetime.now(),
            'intent': intent,
            'score': score
        })

        slots = grounded.get('slots', {})
        for key, value in slots.items():
            if value:  #update non empty values
                ctx['current_slots'][key] = value
        
        validation = grounded['validation']
        query_params = grounded['query_params']

        #low confidence handling
        if score < 0.60:
            self.sessions[session_id] = ctx
            return "low_confidence", {'intent':intent, 'confidence': score, 'query': query}
        
        if not validation['is_valid']:
            resolved_params = self._resolve_from_context(
                ctx, 
                query_params,
                validation['missing_slots']
            )
            still_missing = [
                slot for slot in validation['missing_slots'] 
                if slot not in resolved_params.get('resolved_slots', [])
            ]
            if still_missing:

                self.sessions[session_id] = ctx
                return "need_clarification", {
                    'missing_slots': still_missing,
                    'current_params': resolved_params,
                    'intent': intent
                }
            else:
                query_params = resolved_params
                validation['is_valid'] = True

        ctx['intent'] = intent
        ctx['query_params'] = query_params
        self.sessions[session_id] = ctx

        return "ready_to_execute", {
            'intent': intent,
            'score': score,
            'matches': matches,
            'validation': validation,
            'query_params': query_params,
            'context_used': query_params.get('context_sources', {})
        }
        


    def _get_session_context(self, session_id: str) -> Dict:
        """
        Get or create session context for a user.
        
        - session_id: Unique identifier
        - created_at: Timestamp of session creation
        - intent: Current predicted intent
        - current_slots: Latest known values (location, pollutant, etc.)
        - query_params: Parameters for current query
        - query_history: Past successful queries
        - last_successful_query: Most recent complete query
        - messages: Full conversation transcript
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'session_id': session_id,
                'created_at': datetime.now(),
                'intent': None,
                'current_slots': {},
                'query_params': {},
                'messages': [],
                'query_history': [],
                'last_successful_query': None
            }
        return self.sessions[session_id]
    
    def _resolve_from_context(
            self, 
            ctx: Dict, 
            query_params: Dict, 
            missing_slots: List[str]
        ) -> Dict[str, Any]:
        """
        Attempt to fill missing slots using session context.
        
        Arguments:
        - ctx: Session context
        - query_params: Current query parameters
        - missing_slots: Slots that need to be filled
        
        Returns:
        returns: Updated query_params with resolved slots and context sources.
        """

        resolved = query_params.copy()
        resolved['context_sources'] = {} #tracking of values
        resolved['resolved_slots'] = [] #Tracking slots that we filled


        slot_mapping = {
            'pollutant': 'pollutant_type',
            'location' : 'location',
            'time' : 'time_filter',
            'statistic' : 'aggregation'
        }

        for slot in missing_slots:
            resolved_value = None
            source = None
            param_key = slot_mapping.get(slot,slot)

            #1. Check current session slots
            if slot in ctx['current_slots'] and ctx['current_slots'][slot]:
                slot_value = ctx['current_slots'][slot]

                if slot == 'pollutant' and isinstance(slot_value, dict):
                    if 'type' in slot_value:
                        resolved['pollutant_type'] = slot_value['type']
                        resolved_value = slot_value['type']
                        source = 'current_conversation'

                elif slot == 'location' and isinstance(slot_value, dict):
                    if 'name' in slot_value:
                        resolved['location'] = slot_value['name']
                        resolved['location_type'] = slot_value.get('type','unknown')
                        resolved_value = slot_value['name']
                        source = 'current_conversation'
                
                elif slot == 'time':
                    resolved['time_filter'] = slot_value
                    if isinstance(slot_value,dict):
                        resolved['temporal_type'] = slot_value.get('type', 'unknown')
                    resolved_value = slot_value
                    source = 'current_conversation'
                
                elif slot == 'statistic' and isinstance(slot_value,dict):
                    if 'type' in slot_value:
                        resolved['aggregation'] = slot_value['type']
                        resolved_value = slot_value['type']
                        source = 'current_conversation'

            #2. Check in last successful query
            if not resolved_value and ctx.get('last_successful_query'):
                last_query = ctx['last_successful_query']

                if param_key in last_query:
                    resolved[param_key] = last_query[param_key]
                    resolved_value = last_query[param_key]
                    source = 'previous_query'

                    #copy parameters
                    if param_key == 'location' and 'location_type' in last_query:
                        resolved['location_type'] = last_query['location_type']
                    elif param_key == 'time_filter' and 'temporal_type' in last_query:
                        resolved['temporal_type'] = last_query['temporal_type']
            
            #3. use defaults
            if not resolved_value:
                defaults = self._get_slot_defaults(slot,ctx)
                if defaults:
                    resolved.update(defaults)
                    resolved_value = defaults.get(param_key)
                    source = 'default'

            #track what is resolved
            if resolved_value:
                resolved['resolved_slots'].append(slot)
                resolved['context_sources'][slot] = source
        
        return resolved
    
    def _get_slot_defaults(self, slot: str, ctx: Dict) -> Optional[Dict]:
        """
        Get defaults for slots

        - time: "Today"
        - aggregation: "mean"
        - location/pollutant: most recent
        """
        defaults = {}

        if slot == 'time':
            defaults['time_filter'] = {
                'parsed': 'today',
                'type': 'relative',
                'raw_text': 'today'
            }
        elif slot== 'statistic':
            defaults['aggregation'] = 'mean'
        
        if ctx.get('query_history'):
            history = ctx['query_history']

            if slot == 'location' and len(history) > 0:
                for query in reversed(history):
                    params = query.get('params',{})
                    if 'location' in params:
                        defaults['location'] = params['location']
                        defaults['location_type'] = params.get('location_type','city')
                        break

            elif slot == 'pollutant' and len(history) > 0:
                for query in reversed(history):
                    param = query.get('params',{})
                    if 'pollutant_type' in params:
                        defaults['pollutant_type'] = params['pollutant_type']
                        break
            
        
        return defaults if defaults else None
    
    def update_after_successful_query(
            self,
            session_id: str,
            query_params: Dict[str, Any],
            result: Dict[str,Any]
    ):
        """
        Save a successful query to session history

        Args:
            session_id: session identifier
            query_params: The parameters used for the query
            result: The query result(value, unit, etc,)
        """

        ctx = self.sessions.get(session_id)
        if not ctx:
            return
        
        ctx['last_successful_query'] = query_params.copy()

        ctx['query_history'].append({
            'params': query_params,
            'result': result,
            'timestamp' : datetime.now()
        })

        if len(ctx['query_history'])> 10:
            ctx['query_history'] = ctx['query_history'][-10]

    
    def get_query_history(self, session_id: str) -> List[Dict]:
        """
        Get query history for a sesssion         

        Args:
            session_id: session identifier

        Returns:
            List of dicts with 'params', 'result' and 'timestamp'
        """

        ctx = self.sessions(session_id)
        return ctx['query_history'] if ctx else []
    
    def clear_session(self,session_id:str):
        """
        Clear context

        Args:
            session_id: Session identifier
        """

        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_session_info(self, session_id:str) -> Optional[dict]:
        """
        Get session info

        Args:
            session_id: Session Identifier

        Returns:
            Session context dict or None if session doesn't exist
        """
        return self.sessions.get(session_id)