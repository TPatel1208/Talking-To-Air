import matplotlib.pyplot as plt
import xarray as xr
import time 

from utils import plotting
from utils.statistics import compute_statistic, daily_cycle_peak, get_time_index_local
from utils.plotting import mask_data_by_geometry


from config.intents import INTENT_EXAMPLES

from nlp.intent_embedder import IntentEmbedder
from nlp.slot_extractor import SlotExtractor
from nlp.query_grounder import QueryGrounder
from nlp.dialouge_manager import Dialouge_Manager
from nlp.response_generator import ResponseGenerator
from nlp.query_executor import QueryExecutor

#Loading Data and Initializing Components
print("Loading TEMPO NO2 dataset...")
#ds_TEMPO_NO2= xr.open_dataset('TEMPO_NO2_L3_Warm_Season_Mean_V3.nc4')['NO2_trop_column_good']

ds_TEMPO_NO2_1 = xr.open_dataset('TEMPO_NO2_L3_V03_20250916T210309Z_S012.nc')
no2_grid = ds_TEMPO_NO2_1['weight'] 

resolver = plotting.RegionResolver()
embedder = IntentEmbedder(INTENT_EXAMPLES)
slot_extractor = SlotExtractor()
grounder = QueryGrounder(slot_extractor)
dm = Dialouge_Manager(intent_embedder=embedder, slot_extractor=slot_extractor, query_grounder=grounder)
rg = ResponseGenerator()
qe = QueryExecutor(resolver, default_ds=no2_grid)

session_id = "test_user_123"
query = None
while query != "exit":
    query = input("Enter your query ('exit' to quit): ").strip()
    start = time.time()


    status, data = dm.process_query(session_id, query)
    print(f"Status: {status}")
    print(f"Intent: {data['intent']}")

    # Handle response based on status
    if status == 'ready_to_execute':
        # Complete - generate confirmation
        print(f"Query params: {data['query_params']}")
        print(data['query_params'])
        confirmation = rg.generate_confirmation(
            intent=data['intent'],
            query_params=data['query_params'],
            context_used=data.get('context_used', {})
        )
        print(f"Confirmation: {confirmation}")
        
    elif status == 'need_clarification':
        # Missing info - ask user
        response = rg.generate_clarification(
            missing_slots=data['missing_slots'],
            current_params=data['current_params']
        )
        print(f"Response: {response}")
        
    elif status == 'low_confidence':
        # Unclear intent
        response = rg.generate_low_confidence(data['confidence'])
        print(f"Response: {response}")
    response = qe.execute_query(data)
    if not response:
        print("Could not execute query. Please try again.")
        continue
    elif response.get('PLOT'):
        fig, ax = response['PLOT']
        plt.show()
    elif response.get('TEMPORAL'):
        print(f"Response: {response}")
    elif response.get('STATISTIC'):
        print(f"Response: {response}")

    end = time.time()
    print(f"Query executed in {end - start:.2f} seconds.")
