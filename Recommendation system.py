from distutils.command.install import install


pip install lightfm

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
import numpy as np

# Sample data (user, item, rating)
data = [
    ('User1', 'Item1', 5),
    ('User1', 'Item2', 4),
    ('User2', 'Item1', 3),
    ('User2', 'Item2', 5),
    ('User3', 'Item2', 2),
]

# Create a LightFM dataset
dataset = Dataset()
dataset.fit(users=(user[0] for user, _, _ in data), items=(item[1] for _, item, _ in data))

# Build the interaction matrix
(interactions, weights) = dataset.build_interactions((user[0], item[1], rating) for user, item, rating in data)

# Create and train the model
model = LightFM(loss='warp')
model.fit(interactions, epochs=30, num_threads=2)

# Get recommendations for a user
user_id = 'User1'
n_users, n_items = interactions.shape
known_positives = np.array([item[1] for item, rating in zip(data, interactions.tocsr()[dataset.to_inner_user_ids(user_id)]) if rating > 0])

# Make recommendations
scores = model.predict(dataset.to_inner_user_ids(user_id), np.arange(n_items))
top_items = np.argsort(-scores)

# Print top recommendations
print(f"Top recommendations for {user_id}:")
for item_id in top_items:
    if dataset.to_raw_item_ids(item_id) not in known_positives:
        print(f"Item: {dataset.to_raw_item_ids(item_id)}, Score: {scores[item_id]}")
