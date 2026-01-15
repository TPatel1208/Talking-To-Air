INTENT_EXAMPLES = {

    "MAP_PLOT": [

        # Canonical map requests
        "show me a map of NO2",
        "display NO2 map",
        "plot NO2 over a region",
        "show NO2 spatial distribution",
        "visualize NO2 concentrations",

        # Implicit map language
        "NO2 over California",
        "NO2 across Los Angeles",
        "how does NO2 look over the city",
        "map the NO2 levels in New Jersey",

        # Time-qualified maps (edge cases)
        "show a map of NO2 over LA at 10 AM",
        "plot NO2 over California at noon",
        "display NO2 map for New Jersey yesterday",
        "show NO2 spatial distribution this morning",

        # Temporal aggregation maps
        "show average NO2 map over California today",
        "plot mean NO2 over LA this week",
        "display median NO2 spatial distribution over the last 24 hours",

        # Multi-region maps
        "plot NO2 in New Jersey and California",
        "show NO2 over LA and San Francisco",
        "compare NO2 maps for California and Nevada",
        "side by side NO2 maps for Texas and Arizona"
    ],


    "STATISTIC": [

        # Canonical statistics
        "what is the maximum NO2 level",
        "find the minimum NO2",
        "what is the average NO2",
        "what is the median NO2 concentration",

        # Implicit numeric language (edge cases)
        "how high is NO2 in LA",
        "how bad is NO2 today",
        "NO2 level in the city",
        "current NO2 concentration",
        "is NO2 high or low",

        # Time-qualified statistics
        "what is the maximum NO2 in LA at 10 AM",
        "average NO2 in California at noon",
        "minimum NO2 in New Jersey at 3 PM",

        # Temporal aggregation statistics
        "average NO2 today",
        "mean NO2 in California this week",
        "median NO2 over the last 24 hours",
        "highest NO2 level observed yesterday",

        # Multiple statistics (edge cases)
        "NO2 statistics for a region",
        "give me NO2 min max and average",
        "summarize NO2 levels in LA",
        "show NO2 min median and max for New Jersey",

        # Multi-region statistics
        "compare average NO2 in LA and SF",
        "what is the maximum NO2 in New Jersey and California",
        "show mean NO2 for multiple cities today"
    ],


    "TEMPORAL_ANALYSIS": [

        # Canonical temporal extrema
        "when is NO2 highest",
        "when do we see the lowest NO2 levels",
        "time of day with maximum NO2",
        "NO2 peak time of day",

        # Time-qualified extrema (edge cases)
        "when was NO2 highest today",
        "what time did NO2 peak in LA yesterday",
        "when did NO2 reach its minimum in California",

        # Temporal pattern analysis
        "how does NO2 change over the day",
        "what is the daily NO2 pattern in LA",
        "does NO2 increase in the afternoon",
        "how does NO2 vary throughout the week",

        # Multi-region temporal analysis
        "when is NO2 highest in New Jersey and California",
        "compare NO2 peak times in LA and SF",
        "what time do multiple cities see their highest NO2"
    ]
}
