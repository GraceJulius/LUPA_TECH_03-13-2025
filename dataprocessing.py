import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_product_category(product_data: Dict) -> str:
    """Enhanced category detection with VR/AR compatibility check"""
    category = product_data.get('category', 'general')
    # Add VR/AR compatibility flag
    if product_data.get('vr_compatible', False) or product_data.get('ar_compatible', False):
        return f"{category}_xr"
    return category

def calculate_sustainability_impact(interaction: Dict) -> float:
    """Calculate estimated energy savings for sustainable ad targeting"""
    base_energy = 0.5  # kWh per wasted ad impression
    return base_energy * interaction.get('interaction_duration', 0)

def process_data(interactions: List[Dict], products: List[Dict], 
                vr_tryons: List[Dict] = None) -> pd.DataFrame:
    """
    Enhanced data processor with sustainability tracking and VR/AR integration
    
    Args:
        interactions: List of user interaction dictionaries
        products: List of product metadata dictionaries
        vr_tryons: List of VR try-on session data (optional)
    
    Returns:
        pd.DataFrame: Enhanced dataframe with sustainability metrics and XR features
    """
    try:
        # Input validation
        if not interactions or not products:
            raise ValueError("Missing required input data")
            
        logger.info(f"Processing {len(interactions)} interactions and {len(products)} products")
        
        # Create DataFrames with optimized data types
        interactions_df = pd.DataFrame(interactions).astype({
            'user_id': 'category',
            'product_id': 'category',
            'interaction_type': 'category'
        })
        
        products_df = pd.DataFrame(products).astype({
            'product_id': 'category',
            'category': 'category'
        })
        
        # Merge datasets
        df = pd.merge(interactions_df, products_df, on='product_id', how='left')
        
        # Add VR/AR features
        if vr_tryons:
            vr_df = pd.DataFrame(vr_tryons).astype({
                'user_id': 'category',
                'product_id': 'category'
            })
            df = pd.merge(df, vr_df, on=['user_id', 'product_id'], how='left')
            df['vr_engaged'] = df['vr_duration'] > 0
        
        # Feature engineering
        df['category'] = df.apply(lambda x: get_product_category(x), axis=1)
        df['sustainability_impact'] = df.apply(calculate_sustainability_impact, axis=1)
        
        # Time-based features
        df['interaction_date'] = pd.to_datetime(df['timestamp'])
        df['day_of_week'] = df['interaction_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Enhanced cleaning
        df = df.dropna(subset=['product_id', 'user_id'])
        df = df.drop_duplicates(subset=['user_id', 'product_id', 'timestamp'])
        
        # Memory optimization
        df = df.astype({
            'price': 'float32',
            'sustainability_impact': 'float32'
        })
        
        logger.info(f"Processed dataset shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise

# Example usage with VR data
products = [
    {"product_id": "elec001", "name": "Smartphone", "price": 699, 
     "vr_compatible": True, "category": "electronics"},
    {"product_id": "fash001", "name": "VR Jacket", "price": 199, 
     "ar_compatible": True, "category": "fashion"}
]

interactions = [
    {"user_id": "u123", "product_id": "elec001", "interaction_type": "vr_tryon", 
     "timestamp": "2023-08-20 14:30:00", "interaction_duration": 15},
    {"user_id": "u123", "product_id": "fash001", "interaction_type": "ar_preview", 
     "timestamp": "2023-08-20 15:00:00", "interaction_duration": 8}
]

vr_tryons = [
    {"user_id": "u123", "product_id": "elec001", "vr_duration": 15, 
     "vr_engagement": 0.85}
]

processed_df = process_data(interactions, products, vr_tryons)
print("\nEnhanced Processed Data:")
print(processed_df[['user_id', 'product_id', 'category', 'sustainability_impact', 'vr_engaged']])

def process_data_with_retry(interactions, products, vr_tryons=None, max_retries=3):
    """Process data with retry logic for resilience"""
    for attempt in range(max_retries):
        try:
            return process_data(interactions, products, vr_tryons)
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error("All retry attempts failed")
                raise


def validate_input_data(interactions, products):
    """Validate input data structure before processing"""
    required_interaction_fields = ['user_id', 'product_id', 'interaction_type', 'timestamp']
    required_product_fields = ['product_id', 'name', 'price', 'category']
    
    for interaction in interactions:
        for field in required_interaction_fields:
            if field not in interaction:
                raise ValueError(f"Missing required field '{field}' in interaction data")
    
    for product in products:
        for field in required_product_fields:
            if field not in product:
                raise ValueError(f"Missing required field '{field}' in product data")
    
    return True


import functools

@functools.lru_cache(maxsize=128)
def get_product_details(product_id):
    """Cached function to get product details"""
    # In a real system, this might query a database
    return next((p for p in products if p['product_id'] == product_id), None)

def track_metrics(df):
    """Track key metrics from processed data"""
    metrics = {
        'total_interactions': len(df),
        'unique_users': df['user_id'].nunique(),
        'unique_products': df['product_id'].nunique(),
        'vr_engagement_rate': df['vr_engaged'].mean() if 'vr_engaged' in df.columns else 0,
        'avg_sustainability_impact': df['sustainability_impact'].mean(),
        'weekend_interaction_rate': df[df['is_weekend']].shape[0] / df.shape[0]
    }
    
    logger.info(f"Metrics: {metrics}")
    return metrics


def export_processed_data(df, output_path=None):
    """Export processed data to various formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_path:
        # CSV export
        csv_path = f"{output_path}/processed_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Data exported to CSV: {csv_path}")
        
        # Parquet export (more efficient for large datasets)
        parquet_path = f"{output_path}/processed_data_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Data exported to Parquet: {parquet_path}")
    
    return {
        'csv': csv_path if output_path else None,
        'parquet': parquet_path if output_path else None
    }

def process_incremental_data(new_interactions, new_products, existing_df=None):
    """Process only new data and merge with existing processed data"""
    if existing_df is not None:
        # Get latest timestamp from existing data
        latest_ts = existing_df['interaction_date'].max()
        
        # Filter only new interactions
        new_interactions = [i for i in new_interactions 
                            if pd.to_datetime(i['timestamp']) > latest_ts]
        
        # Process new data
        if new_interactions:
            new_df = process_data(new_interactions, new_products)
            # Concatenate with existing data
            return pd.concat([existing_df, new_df]).drop_duplicates()
        return existing_df
    else:
        # No existing data, process everything
        return process_data(new_interactions, new_products)
    


