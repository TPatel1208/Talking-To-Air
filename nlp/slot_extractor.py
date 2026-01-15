import re
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

class TrieNode:
    def __init__(self):
        self.children = {}
        self.output = []

class KeywordMatching:
    def __init__ (self):
        self.root = TrieNode()
    def add_keyword(self, keyword: str, slot_type:str, value: str):
        """Add a keyword to the trie"""
        node = self.root
        for char in keyword.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.output.append((slot_type,value))
    
    def search(self, text:str) -> List[Tuple[int,int,str,str]]:
        """
        Finds alll keywords that match in text
        Returns (start, end, slot_type, value)
        """
        text_lower = text.lower()
        matches = []

        for start_idx in range(len(text_lower)):
            node = self.root
            for end_idx in range(start_idx, len(text_lower)):
                char = text_lower[end_idx]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.output:
                    for slot_type, value in node.output:
                        matches.append((start_idx, end_idx+1,slot_type,value))
        
        return matches
    


class SlotExtractor:
    def __init__ (self):
        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.keyword_matcher = KeywordMatching()

        #Known Pollutants  
        self.pollutants = {
            'pm2.5': ['pm2.5', 'pm 2.5', 'fine particulate matter', 'fine particles'],
            'pm10': ['pm10', 'pm 10', 'coarse particulate matter', 'coarse particles'],
            'o3': ['o3', 'ozone', 'ground-level ozone'],
            'no2': ['no2', 'nitrogen dioxide'],
            'so2': ['so2', 'sulfur dioxide', 'sulphur dioxide'],
            'co': ['co', 'carbon monoxide'],
        }
        
        # popular location abbreviations
        self.location_abbrev = {
            # Major cities
            'nyc': 'New York City',
            'la': 'Los Angeles',
            'sf': 'San Francisco',
            'dc': 'Washington DC',
            'phx': 'Phoenix',
            'chi': 'Chicago',
            'atl': 'Atlanta',
            'sea': 'Seattle',
            'bos': 'Boston',
            'phi': 'Philadelphia',

            'sd': 'San Diego',
            'sj': 'San Jose',
            'oak': 'Oakland',
            'sac': 'Sacramento',
            'lv': 'Las Vegas',
            'reno': 'Reno',
            'den': 'Denver',
            'slc': 'Salt Lake City',

            'dal': 'Dallas',
            'fw': 'Fort Worth',
            'hou': 'Houston',
            'aus': 'Austin',
            'sa': 'San Antonio',

            'mia': 'Miami',
            'orl': 'Orlando',
            'tpa': 'Tampa',
            'jax': 'Jacksonville',

            'det': 'Detroit',
            'cle': 'Cleveland',
            'cin': 'Cincinnati',
            'col': 'Columbus',

            'minn': 'Minneapolis',
            'stp': 'Saint Paul',
            'mil': 'Milwaukee',

            'kc': 'Kansas City',
            'stl': 'St. Louis',

            'nola': 'New Orleans',
            'baton': 'Baton Rouge',

            'pdx': 'Portland',
            'eug': 'Eugene',

            # Airport-style aliases users commonly type
            'lax': 'Los Angeles',
            'sfo': 'San Francisco',
            'sjc': 'San Jose',
            'smf': 'Sacramento',
            'jfk': 'New York City',

            # States (low-ambiguity only)
            'ca': 'California',
            'tx': 'Texas',
            'ny': 'New York',
            'fl': 'Florida',
            'il': 'Illinois',
            'pa': 'Pennsylvania',
            'oh': 'Ohio',
            'ga': 'Georgia',
            'nc': 'North Carolina',
            'sc': 'South Carolina',
            'va': 'Virginia',
            'wa': 'Washington',
            'az': 'Arizona',
            'co': 'Colorado',
            'ut': 'Utah',
            'nv': 'Nevada',
            'mn': 'Minnesota',
            'wi': 'Wisconsin',
            'mi': 'Michigan',
            'mo': 'Missouri',
            'tn': 'Tennessee',
            'ky': 'Kentucky',
            'al': 'Alabama',
            'ms': 'Mississippi',
            'la-state': 'Louisiana', 
        }
        self.statistics = {
            'max': ['highest', 'maximum', 'max', 'peak', 'top', 'greatest', 'most'],
            'min': ['lowest', 'minimum', 'min', 'bottom', 'least', 'smallest'],
            'mean': ['average', 'mean', 'avg', 'typical'],
            'median': ['median', 'middle', 'mid'],
        }
        self.comparison_keywords = ['compare', 'versus', 'vs', 'vs.', 'difference between', 'compared to']

        #adds keywords to keyword matching
        self._build_keyword_matcher()

        #for regex patterns 
        self._compile_patterns()

    def _build_keyword_matcher(self):
        """Adds known patterns to keyword matching"""

        for pollutant, synonyms in self.pollutants.items(): #pollutants
            for synonym in synonyms:
                self.keyword_matcher.add_keyword(synonym, 'pollutant', pollutant)

        for abbrev, full_name in self.location_abbrev.items(): #location abbreviations
            self.keyword_matcher.add_keyword(abbrev, 'location_abbrev', full_name)
        
        for stat_type, synonyms in self.statistics.items(): #statistics
            for synonym in synonyms:
                self.keyword_matcher.add_keyword(synonym, 'statistic', stat_type)
        
        for keyword in self.comparison_keywords: #comparison
            self.keyword_matcher.add_keyword(keyword, 'comparison', 'comparison')

    def _compile_patterns(self):
        """regex patterns"""
        self.time_pattern = re.compile(
            r'(?P<specific_time>\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)\b)|'
            r'(?P<date>\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|'
            r'(?P<relative_day>\b(?:today|yesterday|tomorrow)\b)|'
            r'(?P<relative_time>\b(?:last|past|previous|next)\s+(?:\d+\s+)?(?:hour|day|week|month|year)s?\b)|'
            r'(?P<day_of_week>\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)|'
            r'(?P<month>\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b)',
            re.IGNORECASE
        )
        
        self.hour_pattern = re.compile(
            r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?',
            re.IGNORECASE
        )

    def extract_slots(self, query: str, intent:str) ->  Dict[str, Any]:
        """process once with spaCY, then extract slots"""
        doc = self.nlp(query)
        query_lower = query.lower()
        keyword_matches = self.keyword_matcher.search(query)

        slots = {
            'pollutant': None,
            'location': None,
            'statistic': None,
            'time': None,
            'comparison': None,
        }

        slots = self._process_keyword_matches(keyword_matches, slots, query)
        #Extract  from spacy
        if not slots['location']:
            slots['location'] = self._extract_location_from_entities(doc)
        #extract from time patterns
        if not slots['time']:
            slots['time'] = self._extract_time_optimized(query, query_lower)
        
        # Calculate confidence
        slots['confidence'] = self._calculate_confidence(slots)

        return slots
        
    def _process_keyword_matches(self, matches: List[Tuple], slots: Dict, query: str) -> Dict:
        """Process all keyword matches found in pass"""
        matches.sort(key = lambda x: (x[0],-(x[1]-x[0])))
        used_ranges = []

        for start, end, slot_type, value in matches:
            #this is used to check overlap so if already used in range skipped
            overlap = any(
                (start < used_end and end > used_start)
                for used_start, used_end in used_ranges
            )
            if overlap:
                continue
            if slot_type == 'pollutant' and not slots['pollutant']:
                    slots['pollutant'] = {
                        'type': value,
                        'raw_text': query[start:end],
                        'confidence': 0.9
                    }
                    used_ranges.append((start, end))
            
            elif slot_type == 'location_abbrev' and not slots['location']:
                slots['location'] = {
                    'name': value,
                    'raw_text': query[start:end],
                    'type': 'city',
                    'confidence': 0.95
                }
                used_ranges.append((start, end))
            
            elif slot_type == 'statistic' and not slots['statistic']:
                slots['statistic'] = {
                    'type': value,
                    'raw_text': query[start:end],
                    'confidence': 0.9
                }
                used_ranges.append((start, end))
            
            elif slot_type == 'comparison' and not slots['comparison']:
                slots['comparison'] = {
                    'type': 'comparison',
                    'raw_text': query[start:end],
                    'confidence': 0.9
                }
                used_ranges.append((start, end))
        
        return slots
    def _extract_location_from_entities(self, doc) -> Optional[Dict[str, Any]]:
        """Use spacy to extract location"""
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:
                return {
                    'name': ent.text,
                    'raw_text': ent.text,
                    'type': ent.label_.lower(),
                    'confidence': 0.90
                }
        return None
    

    def _extract_time_optimized(self, query: str, query_lower: str) -> Optional[Dict[str, Any]]:
        """Extract temporal information using pre-compiled patterns"""
        #regex search with all patterns
        match = self.time_pattern.search(query_lower)
        
        if not match:
            return None
        
        time_info = {
            'raw_text': match.group(0),
            'parsed': None,
            'type': None,
            'confidence': 0.0
        }
        
        # Determine which pattern matched
        if match.lastgroup == 'specific_time':
            # Parse hour with AM/PM
            hour_match = self.hour_pattern.search(match.group(0))
            if hour_match:
                hour = int(hour_match.group(1))
                minute = int(hour_match.group(2)) if hour_match.group(2) else 0
                meridiem = hour_match.group(3)
                
                if meridiem:
                    meridiem = meridiem.replace('.', '').lower()
                    if meridiem == 'pm' and hour != 12:
                        hour += 12
                    elif meridiem == 'am' and hour == 12:
                        hour = 0
                
                time_info.update({
                    'parsed': f"{hour:02d}:{minute:02d}",
                    'type': 'specific_time',
                    'hour': hour,
                    'minute': minute,
                    'meridiem': meridiem if meridiem else None,
                    'confidence': 0.95
                })
        
        elif match.lastgroup == 'date':
            parts = re.split(r'[/-]', match.group(0))
            month, day, year = parts
            if len(year) == 2:
                year = f"20{year}"
            
            time_info.update({
                'parsed': f"{year}-{int(month):02d}-{int(day):02d}",
                'type': 'specific_date',
                'confidence': 0.95
            })
        
        elif match.lastgroup == 'relative_day':
            time_info.update({
                'parsed': match.group(0),
                'type': 'relative_day',
                'confidence': 0.9
            })
        
        elif match.lastgroup == 'relative_time':
            text = match.group(0)
            direction = re.search(r'(last|past|previous|next)', text).group(1)
            amount_match = re.search(r'\d+', text)
            amount = int(amount_match.group(0)) if amount_match else 1
            unit = re.search(r'(hour|day|week|month|year)', text).group(1)
            
            time_info.update({
                'parsed': f"{direction} {amount} {unit}",
                'type': 'relative_period',
                'direction': direction,
                'amount': amount,
                'unit': unit,
                'confidence': 0.9
            })
        
        elif match.lastgroup == 'day_of_week':
            time_info.update({
                'parsed': match.group(0),
                'type': 'day_of_week',
                'confidence': 0.85
            })
        
        elif match.lastgroup == 'month':
            time_info.update({
                'parsed': match.group(0),
                'type': 'month',
                'confidence': 0.85
            })
        
        return time_info if time_info['confidence'] > 0 else None
    
    def _calculate_confidence(self, slots: Dict[str, Any]) -> float:
        """overall confidence score"""
        confidences = []
        
        for slot_name, slot_value in slots.items():
            if slot_name == 'confidence':
                continue
            if slot_value and isinstance(slot_value, dict) and 'confidence' in slot_value:
                confidences.append(slot_value['confidence'])
        
        return np.mean(confidences) if confidences else 0.0
    