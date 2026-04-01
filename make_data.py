import pandas as pd
import os

if not os.path.exists('data'):
    os.makedirs('data')

# 0 = Human, 1 = AI
data = {
    'text': [
        "I went to the store and bought some milk.", 
        "The socioeconomic implications of machine learning are profound and multifaceted.",
        "My favorite car is a Nissan Skyline R34 because of the RB26 engine.",
        "In this essay, we will examine the intricate nuances of global trade dynamics.",
        "I forgot my umbrella today and got soaked in the rain near QCU.",
        "Artificial intelligence represents a paradigm shift in computational linguistics."
    ],
    'generated': [0, 1, 0, 1, 0, 1]
}

pd.DataFrame(data).to_csv('data/train_essays.csv', index=False)
print("✅ Test data created in data/train_essays.csv")