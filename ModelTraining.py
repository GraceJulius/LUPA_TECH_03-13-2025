import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class AmazonWishRecommender:
    """
    Recommendation system for AmazonWish platform that leverages user-product interaction data
    and VR/AR engagement metrics to provide personalized product recommendations.
    """
    
    def __init__(self, n_neighbors=5, metric='cosine'):
        """
        Initialize the AmazonWish recommendation system.
        
        Parameters:
            n_neighbors (int): Number of similar users/items to consider
            metric (str): Distance metric for similarity calculation
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.user_model = None
        self.item_model = None
        self.user_scaler = None
        self.item_scaler = None
        self.user_features = None
        self.item_features = None
        self.user_item_matrix = None
        
    def preprocess_data(self, interaction_data, user_data=None, product_data=None):
        """
        Preprocess interaction data and merge with user and product features.
        
        Parameters:
            interaction_data (DataFrame): User-product interactions with engagement metrics
            user_data (DataFrame): User demographic and behavioral data
            product_data (DataFrame): Product features and metadata
            
        Returns:
            processed_data (DataFrame): Processed data ready for model training
        """
        # Clean and prepare interaction data
        processed_data = interaction_data.copy()
        
        # Fill missing values
        processed_data.fillna({
            'view_time_seconds': processed_data['view_time_seconds'].median(),
            'vr_engagement_score': processed_data['vr_engagement_score'].median(),
            'click_through_rate': processed_data['click_through_rate'].median(),
            'cart_add': False,
            'purchase': False
        }, inplace=True)
        
        # Create engagement score combining multiple metrics
        processed_data['engagement_score'] = (
            0.3 * processed_data['view_time_seconds'] / processed_data['view_time_seconds'].max() +
            0.3 * processed_data['vr_engagement_score'] / 10 +
            0.2 * processed_data['click_through_rate'] +
            0.1 * processed_data['cart_add'].astype(int) +
            0.1 * processed_data['purchase'].astype(int)
        )
        
        # Normalize engagement score
        processed_data['engagement_score'] = processed_data['engagement_score'] / processed_data['engagement_score'].max() * 5
        
        # Merge with user and product data if provided
        if user_data is not None:
            processed_data = processed_data.merge(user_data, on='user_id', how='left')
            
        if product_data is not None:
            processed_data = processed_data.merge(product_data, on='product_id', how='left')
            
        return processed_data
        
    def create_user_item_matrix(self, processed_data):
        """
        Create user-item interaction matrix from processed data.
        
        Parameters:
            processed_data (DataFrame): Processed interaction data
            
        Returns:
            user_item_matrix (DataFrame): Matrix with users as rows, items as columns
        """
        # Create matrix with engagement scores
        user_item_matrix = processed_data.pivot(
            index='user_id', 
            columns='product_id', 
            values='engagement_score'
        )
        
        # Fill NaN values with 0 (no interaction)
        user_item_matrix = user_item_matrix.fillna(0)
        
        return user_item_matrix
        
    def extract_user_features(self, processed_data):
        """
        Extract user features for content-based filtering.
        
        Parameters:
            processed_data (DataFrame): Processed data with user features
            
        Returns:
            user_features (DataFrame): User features for similarity calculation
        """
        # Select relevant user features
        user_cols = [
            'user_id', 'age', 'gender', 'vr_device_owned', 'avg_session_time',
            'price_sensitivity', 'style_preference', 'tech_savviness',
            'fashion_interest', 'sustainability_interest'
        ]
        
        # Filter columns that exist in the data
        available_cols = [col for col in user_cols if col in processed_data.columns]
        
        # Extract unique users with their features
        user_features = processed_data[available_cols].drop_duplicates('user_id').set_index('user_id')
        
        # Handle categorical variables
        if 'gender' in user_features.columns:
            user_features = pd.get_dummies(user_features, columns=['gender'], drop_first=True)
            
        if 'style_preference' in user_features.columns:
            user_features = pd.get_dummies(user_features, columns=['style_preference'], prefix='style')
            
        if 'vr_device_owned' in user_features.columns:
            user_features = pd.get_dummies(user_features, columns=['vr_device_owned'], prefix='device')
        
        return user_features
    
    def extract_item_features(self, processed_data):
        """
        Extract product features for content-based filtering.
        
        Parameters:
            processed_data (DataFrame): Processed data with product features
            
        Returns:
            item_features (DataFrame): Product features for similarity calculation
        """
        # Select relevant product features
        item_cols = [
            'product_id', 'category', 'price', 'brand', 'avg_rating',
            'vr_compatibility_score', 'ar_compatibility_score', 
            'sustainability_score', 'popularity'
        ]
        
        # Filter columns that exist in the data
        available_cols = [col for col in item_cols if col in processed_data.columns]
        
        # Extract unique products with their features
        item_features = processed_data[available_cols].drop_duplicates('product_id').set_index('product_id')
        
        # Handle categorical variables
        if 'category' in item_features.columns:
            item_features = pd.get_dummies(item_features, columns=['category'], prefix='cat')
            
        if 'brand' in item_features.columns:
            item_features = pd.get_dummies(item_features, columns=['brand'], prefix='brand')
        
        return item_features
        
    def fit(self, processed_data):
        """
        Train the recommendation models based on processed data.
        
        Parameters:
            processed_data (DataFrame): Processed interaction and feature data
            
        Returns:
            self: Trained model instance
        """
        # Create user-item matrix
        self.user_item_matrix = self.create_user_item_matrix(processed_data)
        
        # Extract user and item features
        self.user_features = self.extract_user_features(processed_data)
        self.item_features = self.extract_item_features(processed_data)
        
        # Scale user features
        self.user_scaler = StandardScaler()
        scaled_user_features = self.user_scaler.fit_transform(self.user_features)
        
        # Scale item features
        self.item_scaler = StandardScaler()
        scaled_item_features = self.item_scaler.fit_transform(self.item_features)
        
        # Train user-based model
        self.user_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.user_features)),
            algorithm='auto',
            metric=self.metric
        )
        self.user_model.fit(scaled_user_features)
        
        # Train item-based model
        self.item_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.item_features)),
            algorithm='auto',
            metric=self.metric
        )
        self.item_model.fit(scaled_item_features)
        
        return self
    
    def recommend_products_for_user(self, user_id, n_recommendations=5, include_vr_score=True):
        """
        Generate product recommendations for a specific user.
        
        Parameters:
            user_id: ID of the user
            n_recommendations (int): Number of recommendations to generate
            include_vr_score (bool): Whether to factor in VR compatibility
            
        Returns:
            recommendations (DataFrame): Recommended products with scores
        """
        if user_id not in self.user_features.index:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        # Get user features and scale
        user_idx = self.user_features.index.get_loc(user_id)
        user_vector = self.user_features.iloc[user_idx].values.reshape(1, -1)
        scaled_user_vector = self.user_scaler.transform(user_vector)
        
        # Find similar users
        distances, indices = self.user_model.kneighbors(scaled_user_vector)
        similar_users_indices = indices[0][1:]  # Exclude the user itself
        
        # Get user rating vector
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get products the user hasn't interacted with
        user_products = user_ratings[user_ratings > 0].index
        all_products = self.user_item_matrix.columns
        candidate_products = np.setdiff1d(all_products, user_products)
        
        if len(candidate_products) == 0:
            return pd.DataFrame(columns=['product_id', 'score'])
            
        # Get ratings from similar users for candidate products
        similar_users = self.user_item_matrix.iloc[similar_users_indices]
        candidate_ratings = similar_users[candidate_products]
        
        # Calculate average ratings weighted by similarity (1/distance)
        weights = 1 / (distances[0][1:] + 0.1)  # Add small constant to avoid division by zero
        weighted_ratings = pd.DataFrame(
            (candidate_ratings.T * weights).T.mean(),
            columns=['cf_score']
        )
        
        # Add VR/AR compatibility score if available and requested
        if include_vr_score and 'vr_compatibility_score' in self.item_features.columns:
            vr_scores = self.item_features.loc[candidate_products, 'vr_compatibility_score']
            weighted_ratings['vr_score'] = vr_scores / vr_scores.max() * 5
            
            # Combine scores (70% collaborative filtering, 30% VR compatibility)
            weighted_ratings['score'] = 0.7 * weighted_ratings['cf_score'] + 0.3 * weighted_ratings['vr_score']
        else:
            weighted_ratings['score'] = weighted_ratings['cf_score']
            
        # Get top recommendations
        recommendations = weighted_ratings.sort_values('score', ascending=False).head(n_recommendations)
        recommendations.reset_index(inplace=True)
        recommendations.rename(columns={'index': 'product_id'}, inplace=True)
        
        return recommendations
    
    def recommend_similar_products(self, product_id, n_recommendations=5):
        """
        Find products similar to a given product.
        
        Parameters:
            product_id: ID of the reference product
            n_recommendations (int): Number of similar products to recommend
            
        Returns:
            similar_products (DataFrame): Similar products with similarity scores
        """
        if product_id not in self.item_features.index:
            raise ValueError(f"Product ID {product_id} not found in training data")
            
        # Get product features and scale
        item_idx = self.item_features.index.get_loc(product_id)
        item_vector = self.item_features.iloc[item_idx].values.reshape(1, -1)
        scaled_item_vector = self.item_scaler.transform(item_vector)
        
        # Find similar products
        distances, indices = self.item_model.kneighbors(scaled_item_vector)
        
        # Convert to dataframe (exclude the product itself)
        similar_products = pd.DataFrame({
            'product_id': self.item_features.index[indices[0][1:n_recommendations+1]],
            'similarity_score': 1 - distances[0][1:n_recommendations+1]  # Convert distance to similarity
        })
        
        return similar_products
    
    def recommend_for_vr_experience(self, user_id, vr_category=None, n_recommendations=5):
        """
        Generate recommendations optimized for VR experience.
        
        Parameters:
            user_id: ID of the user
            vr_category (str): Category to focus on for VR experience
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            vr_recommendations (DataFrame): VR-optimized product recommendations
        """
        # Get basic recommendations
        base_recommendations = self.recommend_products_for_user(
            user_id, 
            n_recommendations=n_recommendations*2,  # Get more to filter
            include_vr_score=True
        )
        
        # Filter by category if specified
        if vr_category and 'category' in self.item_features.columns:
            product_categories = self.item_features['category'].reindex(base_recommendations['product_id'])
            category_mask = product_categories == vr_category
            if category_mask.any():
                base_recommendations = base_recommendations[category_mask.values]
                
        # Sort by VR compatibility score if available
        if 'vr_compatibility_score' in self.item_features.columns:
            vr_scores = self.item_features['vr_compatibility_score'].reindex(base_recommendations['product_id'])
            base_recommendations['vr_score'] = vr_scores.values
            base_recommendations = base_recommendations.sort_values('vr_score', ascending=False)
            
        # Return top recommendations
        return base_recommendations.head(n_recommendations)
    
    def evaluate(self, test_data, top_n=5):
        """
        Evaluate recommendation model performance.
        
        Parameters:
            test_data (DataFrame): Test interactions data
            top_n (int): Number of recommendations to consider
            
        Returns:
            metrics (dict): Evaluation metrics
        """
        hits = 0
        precision_sum = 0
        recall_sum = 0
        total_users = 0
        
        # Process test data
        processed_test = self.preprocess_data(test_data)
        
        # Group test data by user
        test_user_items = processed_test.groupby('user_id')['product_id'].apply(list)
        
        # Evaluate for each user
        for user_id, actual_items in test_user_items.items():
            if user_id not in self.user_features.index:
                continue
                
            # Get recommendations
            try:
                recommendations = self.recommend_products_for_user(
                    user_id, 
                    n_recommendations=top_n
                )
                recommended_items = recommendations['product_id'].tolist()
                
                # Calculate hits (items that appear in both lists)
                user_hits = len(set(recommended_items) & set(actual_items))
                
                # Calculate precision and recall
                precision = user_hits / len(recommended_items) if recommended_items else 0
                recall = user_hits / len(actual_items) if actual_items else 0
                
                # Update totals
                hits += user_hits
                precision_sum += precision
                recall_sum += recall
                total_users += 1
                
            except Exception as e:
                print(f"Error evaluating user {user_id}: {e}")
                continue
        


        