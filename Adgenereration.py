def generate_ad(recommended_product):
    """Generate more compelling ad creative based on product attributes."""
    product_name = recommended_product['name']
    price = recommended_product.get('price', '')
    category = recommended_product.get('category', '')
    features = recommended_product.get('features', [])
    
    # Different ad templates
    templates = [
        f"Discover the {product_name}—perfect for your needs!",
        f"Love {category}? You'll adore the new {product_name}!",
        f"{product_name}: Quality you deserve, price you'll love",
        f"Just for you: The {product_name} at {price}",
    ]
    
    # Select template based on product attributes
    if features:
        ad_text = f"{random.choice(templates)} Features: {', '.join(features[:2])}"
    else:
        ad_text = random.choice(templates)
        
    return ad_text, recommended_product.get('image_url', "default_image.jpg")

def serve_ad(user_id, ad_variants, ad_image):
    """Serve different ad variants to test effectiveness."""
    variant_id = random.randint(0, len(ad_variants) - 1)
    selected_ad = ad_variants[variant_id]
    
    print(f"Serving ad variant {variant_id} to {user_id}:")
    print("Ad Text:", selected_ad)
    print("Ad Image URL:", ad_image)
    
    # Store which variant was shown to this user
    # track_ad_impression(user_id, variant_id, selected_ad, ad_image)
    
    return variant_id

def generate_personalized_ad(user_id, recommended_product, user_data):
    """Generate ad with personalization based on user history."""
    base_text, ad_image = generate_ad(recommended_product)
    
    # Personalize based on user data
    if 'previous_purchases' in user_data and user_data['previous_purchases']:
        base_text = f"Based on your interest in {user_data['previous_purchases'][-1]}, " + base_text
    
    if 'name' in user_data:
        base_text = f"{user_data['name']}, {base_text}"
        
    return base_text, ad_image

def track_ad_performance(ad_id, user_id, action):
    """Track user interactions with ads."""
    timestamp = datetime.now().isoformat()
    interaction = {
        'ad_id': ad_id,
        'user_id': user_id,
        'action': action,  # 'impression', 'click', 'conversion'
        'timestamp': timestamp
    }
    
    # In production, send to analytics system
    print(f"Tracked {action} for ad {ad_id} by user {user_id}")
    return interaction

def create_ad_campaign(product_ids, budget, targeting_criteria, start_date, end_date):
    """Create a complete ad campaign for multiple products."""
    campaign_id = str(uuid.uuid4())
    
    campaign = {
        'id': campaign_id,
        'product_ids': product_ids,
        'budget': budget,
        'daily_spend_limit': budget / (end_date - start_date).days,
        'targeting': targeting_criteria,
        'start_date': start_date,
        'end_date': end_date,
        'status': 'draft',
        'metrics': {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spend': 0
        }
    }
    
    # In production, store in database
    print(f"Created campaign {campaign_id} for products {product_ids}")
    return campaign

def check_budget_availability(campaign_id, campaigns):
    """Check if campaign has remaining budget before serving an ad."""
    campaign = campaigns.get(campaign_id)
    if not campaign:
        return False
        
    if campaign['metrics']['spend'] >= campaign['budget']:
        return False
        
    current_date = datetime.now().date()
    if current_date < campaign['start_date'] or current_date > campaign['end_date']:
        return False
        
    if campaign['metrics']['spend'] >= campaign['daily_spend_limit']:
        return False
        
    return True

def serve_ad_via_platform(platform, campaign_id, ad_creative, targeting):
    """Integrate with external ad platforms."""
    if platform == 'amazon':
        # Amazon Advertising API integration
        headers = {'Authorization': 'Bearer YOUR_TOKEN'}
        payload = {
            'campaignId': campaign_id,
            'creative': ad_creative,
            'targeting': targeting
        }
        
        # In production, make actual API call
        # response = requests.post('https://advertising-api.amazon.com/v2/ads', json=payload, headers=headers)
        print(f"Ad submitted to Amazon Advertising platform for campaign {campaign_id}")