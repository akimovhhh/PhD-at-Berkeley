from pathlib import Path
import pandas as pd
from typing import List, Tuple

def validate_merge(left_df: pd.DataFrame, right_df: pd.DataFrame, merged_df: pd.DataFrame, merge_key: str) -> None:
    """
    Validate that merge operation didn't lose or add unexpected data.
    
    Args:
        left_df: Left DataFrame in merge
        right_df: Right DataFrame in merge
        merged_df: Resulting merged DataFrame
        merge_key: Key used for merging
    """
    # Check if we lost any rows
    expected_rows = min(left_df[merge_key].nunique(), right_df[merge_key].nunique())
    actual_rows = merged_df[merge_key].nunique()
    
    if actual_rows < expected_rows:
        raise ValueError(f"Merge lost data: Expected at least {expected_rows} unique keys, got {actual_rows}")
    
    # Check for unexpected columns
    expected_columns = set(left_df.columns) | set(right_df.columns)
    actual_columns = set(merged_df.columns)
    
    if expected_columns != actual_columns:
        extra_cols = actual_columns - expected_columns
        missing_cols = expected_columns - actual_columns
        raise ValueError(f"Merge column mismatch. Extra columns: {extra_cols}, Missing columns: {missing_cols}")

def create_city_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create city and state pairs in both direct and non-direct formats.
    
    Args:
        df: DataFrame containing city and state information
    Returns:
        DataFrame with added city and state pair columns
    """
    # Vectorized operations for creating pairs
    pairs = {
        'city_pair_non_direct': df.apply(
            lambda row: str(tuple(sorted([row['Description_x'], row['Description_y']]))), 
            axis=1
        ),
        'state_pair_non_direct': df.apply(
            lambda row: str(tuple(sorted([row['OriginState'], row['DestState']]))), 
            axis=1
        ),
        'city_pair_direct': df.apply(
            lambda row: str(tuple([row['Description_x'], row['Description_y']])), 
            axis=1
        ),
        'state_pair_direct': df.apply(
            lambda row: str(tuple([row['OriginState'], row['DestState']])), 
            axis=1
        )
    }
    
    return pd.concat([df, pd.DataFrame(pairs, index=df.index)], axis=1)

def process_db1b_data(base_path: Path, start_year: int = 2014, end_year: int = 2025) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process DB1B data for Hawaiian routes and Hawaiian Airlines flights.
    
    Args:
        base_path: Path to DB1B data directory
        start_year: Starting year for processing
        end_year: Ending year for processing
    
    Returns:
        Tuple of (Hawaiian routes DataFrame, Hawaiian Airlines DataFrame)
    """
    # Pre-define column sets for efficient reading
    market_cols = ['ItinID', 'TkCarrier', 'OpCarrier', 'MktCoupons', 
                  'OriginCityMarketID', 'DestCityMarketID', 'OriginCountry',
                  'OriginState', 'DestCountry', 'DestState']
    
    ticket_cols = ['ItinID', 'Coupons', 'RoundTrip', 'DollarCred', 'Year',
                  'Quarter', 'ItinFare', 'BulkFare', 'Distance', 'DistanceGroup',
                  'Passengers', 'OnLine']
    
    all_tickets_HI = []
    all_tickets_HA = []
    
    # Process data for each quarter
    for year in range(start_year, end_year):
        for quarter in range(1, 5):
            # Skip future quarters
            if year == 2024 and quarter == 4:
                continue
                
            try:
                # Construct file paths
                ticket_file = base_path / f"Ticket/{year}_{quarter}.csv"
                market_file = base_path / f"Market/Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.csv"
                
                # Read market data with optimized settings
                market = pd.read_csv(
                    market_file,
                    usecols=market_cols,
                    dtype={'ItinID': 'int64', 'MktCoupons': 'int8'}
                )
                
                # Create masks for filtering
                mask_HI = (
                    (market['MktCoupons'] == 1) &
                    ((market['OriginState'] == 'HI') | (market['DestState'] == 'HI'))
                )
                mask_HA = (market['MktCoupons'] == 1) & (market['TkCarrier'] == 'HA')
                
                # Filter market data
                market_HI = market[mask_HI]
                market_HA = market[mask_HA]
                
                # Get unique ItinIDs
                itin_id_HI = set(market_HI['ItinID'])
                itin_id_HA = set(market_HA['ItinID'])
                
                # Read and filter ticket data
                ticket = pd.read_csv(
                    ticket_file,
                    usecols=ticket_cols,
                    dtype={'ItinID': 'int64', 'OnLine': 'int8'}
                )
                
                # Filter tickets
                ticket_HI = ticket[
                    (ticket['ItinID'].isin(itin_id_HI)) &
                    (ticket['OnLine'] == 1)
                ]
                ticket_HA = ticket[
                    (ticket['ItinID'].isin(itin_id_HA)) &
                    (ticket['OnLine'] == 1)
                ]
                
                # Add city names and validate merges
                for df_pair in [(market_HI, ticket_HI), (market_HA, ticket_HA)]:
                    market_df, ticket_df = df_pair
                    
                    # Add origin city
                    market_df = pd.merge(
                        market_df,
                        city_dict,
                        left_on='OriginCityMarketID',
                        right_on='Code',
                        how='left'
                    )
                    validate_merge(market_df, city_dict, market_df, 'OriginCityMarketID')
                    
                    # Add destination city
                    market_df = pd.merge(
                        market_df,
                        city_dict,
                        left_on='DestCityMarketID',
                        right_on='Code',
                        how='left'
                    )
                    validate_merge(market_df, city_dict, market_df, 'DestCityMarketID')
                
                # Create city pairs
                market_HI = create_city_pairs(market_HI)
                market_HA = create_city_pairs(market_HA)
                
                # Select final columns and drop duplicates
                final_cols = ['ItinID', 'TkCarrier', 'city_pair', 'OpCarrier', 'state_pair']
                market_HI = market_HI[final_cols].drop_duplicates()
                market_HA = market_HA[final_cols].drop_duplicates()
                
                # Merge and validate
                merged_HI = pd.merge(market_HI, ticket_HI, on='ItinID')
                merged_HA = pd.merge(market_HA, ticket_HA, on='ItinID')
                validate_merge(market_HI, ticket_HI, merged_HI, 'ItinID')
                validate_merge(market_HA, ticket_HA, merged_HA, 'ItinID')
                
                # Append to results
                all_tickets_HI.append(merged_HI)
                all_tickets_HA.append(merged_HA)
                
            except Exception as e:
                print(f"Error processing {year} Q{quarter}: {str(e)}")
                continue
    
    # Concatenate all results
    df_HI = pd.concat(all_tickets_HI, ignore_index=True)
    df_HA = pd.concat(all_tickets_HA, ignore_index=True)
    
    # Add date column
    for df in [df_HI, df_HA]:
        df['date'] = pd.to_datetime(
            df['Year'].astype(str) + '-' +
            ((df['Quarter'] - 1) * 3 + 1).astype(str) + '-01'
        )
    
    return df_HI, df_HA

# Programm
base_path = Path("/Users/akimovh/Library/CloudStorage/GoogleDrive-akimovhresearch@gmail.com/My Drive/predatory pricing/data/DB1B")
city_dict = pd.read_csv("/Users/akimovh/Library/CloudStorage/GoogleDrive-akimovhresearch@gmail.com/My Drive/predatory pricing/data/dict/L_CITY_MARKET_ID.csv")
df_HI, df_HA = process_db1b_data(base_path)