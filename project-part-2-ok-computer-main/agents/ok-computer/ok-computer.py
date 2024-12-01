import numpy as np
import pickle
from collections import deque  # For rolling window

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']
        self.total_timesteps = params.get('total_timesteps', 2500)

        # Load trained model and transformer
        model_path = "agents/ok-computer/trained_model.pkl"
        try:
            with open(model_path, 'rb') as f:
                loaded_data = pickle.load(f)
                self.trained_model = loaded_data['model']
                self.poly_transformer = loaded_data['poly']
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Rolling window for competitor prices
        self.competitor_prices = deque(maxlen=10)  # Track the last 10 competitor prices

    def _process_last_sale(self, last_sale, inventories):
        # Update inventory
        self.remaining_inventory = inventories[self.this_agent_number]

        # Track competitor's last price
        competitor_price = last_sale[1][1]  # Assuming [self_price, competitor_price]
        self.competitor_prices.append(competitor_price)

    def predict_purchase_probability(self, covariates, price):
        features = np.hstack([covariates, price]).reshape(1, -1)
        features_poly = self.poly_transformer.transform(features)
        probability = self.trained_model.predict_proba(features_poly)[0][1]  # XGBoost predict_proba
        return probability

    def action(self, obs):
        """
        Decide the price to offer to the next customer.
        """
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, inventories)

        if self.remaining_inventory <= 0:
            return 0  # No inventory to sell

        # Dynamic Pricing Strategy
        price_range = np.linspace(1, 100, 50)  # Default price range

        # Adaptive undercutting: Estimate competitor price trend
        avg_competitor_price = np.mean(self.competitor_prices) if len(self.competitor_prices) > 0 else 50  # Default fallback
        undercut_factor = 0.95 if np.std(self.competitor_prices) < 5 else 0.9  # Adjust based on price stability
        competitor_adjusted_price = avg_competitor_price * undercut_factor

        # Inventory-sensitive pricing
        if self.remaining_inventory <= 3:  # Low inventory
            competitor_adjusted_price += 10  # Price conservatively
        elif self.remaining_inventory >= 9:  # High inventory
            competitor_adjusted_price -= 5  # Price aggressively

        # Vectorized probability and revenue computation
        prices = np.array(price_range)
        features = np.hstack([np.tile(new_buyer_covariates, (len(prices), 1)), prices.reshape(-1, 1)])
        features_poly = self.poly_transformer.transform(features)
        probabilities = self.trained_model.predict_proba(features_poly)[:, 1]
        revenues = prices * probabilities

        # Get the best price
        best_price = prices[np.argmax(revenues)]

        # Final adjustment based on competitor pricing
        best_price = max(1, min(best_price, competitor_adjusted_price))  # Ensure price is competitive but reasonable

        # Exploration: Occasionally adjust price randomly
        if np.random.rand() < 0.1:  # 10% chance to explore
            best_price *= np.random.uniform(0.8, 1.2)  # Random adjustment by ±20%

        return best_price