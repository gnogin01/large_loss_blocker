import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def make_address_searchable(df, address_column):
    df_clean = df.copy()
    
    df_clean['searchable_address'] = df_clean[address_column].str.replace(r'([^A-Z])([A-Z])', r'\1 \2', regex = True) 
        # add space before capital letter if previous letter is lower case
        
    df_clean['searchable_address'] = df_clean['searchable_address'].str.replace('NE', 'NE ')
    df_clean['searchable_address'] = df_clean['searchable_address'].str.replace('NW', 'NW ')
    df_clean['searchable_address'] = df_clean['searchable_address'].str.replace('SE', 'SE ')
    df_clean['searchable_address'] = df_clean['searchable_address'].str.replace('SW', 'SW ')
    
    return df_clean


def get_latitude(x):
    if hasattr(x,'latitude') and (x.latitude is not None): 
        return x.latitude


def get_longitude(x):
    if hasattr(x,'longitude') and (x.longitude is not None): 
        return x.longitude
    
    
def get_lat_long_nominatim(df, address_column):
    # NOTE: This function takes FOREVER.
    geolocator = Nominatim(user_agent = "myApp")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds = 0.1)

    df_clean = df.copy()
    df_clean[['latitude_geopy', 'longitude_geopy']] = df_clean[address_column].apply(geocode).apply(lambda x: pd.Series([get_latitude(x), get_longitude(x)], index = ['latitude_geopy', 'longitude_geopy']))

    return df_clean