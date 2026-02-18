import matplotlib.pyplot as plt
import xarray as xr
import time 
import os

from utils import plotting
from utils.statistics import compute_statistic, daily_cycle_peak, get_time_index_local
from utils.plotting import mask_data_by_geometry, RegionResolver


from config.intents import INTENT_EXAMPLES

from nlp.intent_embedder import IntentEmbedder
from nlp.slot_extractor import SlotExtractor
from nlp.query_grounder import QueryGrounder
from nlp.dialouge_manager import Dialouge_Manager
from nlp.response_generator import ResponseGenerator
from nlp.query_executor import QueryExecutor
""
from preprocessing.data_loader import DataLoader

#look zarr compression
"""
data_loader = DataLoader()
files = data_loader.download_file(
    save_dir='data',
    short_name="HAQ_TROPOMI_NO2_GLOBAL_M_L3",
    temporal=("2024-07-01", "2024-07-01"),
    bounding_box=(-125, 25, -66.5, 49.5))  # CONUS bounding box
"""
print("Loading TEMPO NO2 dataset...")
ds_TEMPO_NO2= xr.open_dataset('TEMPO_NO2_L3_Warm_Season_Mean_V3.nc4')['NO2_trop_column_good']
"""path = os.path.join('data', 'HAQ_TROPOMI_NO2_GLOBAL_QA75_L3_Monthly_072024_V2.4_20240810.nc4')
ds_test = xr.open_dataset(path)['Tropospheric_NO2']"""

data_loader = DataLoader()
ds_test = data_loader.get_dataset(
    short_name="HAQ_TROPOMI_NO2_GLOBAL_M_L3",
    temporal=("2024-07-01", "2024-07-01"),
    bounding_box= None
)

da = ds_test['Tropospheric_NO2']


print("Dataset loaded successfully.")
resolver = plotting.RegionResolver()
embedder = IntentEmbedder(INTENT_EXAMPLES)
slot_extractor = SlotExtractor()
grounder = QueryGrounder(slot_extractor)
dm = Dialouge_Manager(intent_embedder=embedder, slot_extractor=slot_extractor, query_grounder=grounder)
rg = ResponseGenerator()
qe = QueryExecutor(resolver, default_ds=da)

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
        print(f"Query params: {data['query_params']} \n")
        response = qe.execute_query(data)
        confirmation = rg.generate_confirmation(
            intent=data['intent'],
            query_params=data['query_params'],
            context_used=data.get('context_used', {})
        )
        print(f"Confirmation: {confirmation}")
        if response.get('PLOT'):
            fig, ax = response['PLOT']
            plt.show()
        elif response.get('TEMPORAL'):
            print(f"Response: {response.values()}")
        elif response.get('STATISTIC'):
            print(f"Response: {response.values()}")

        
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
    
    end = time.time()
    print(f"Query executed in {end - start:.2f} seconds.")
