import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_data(n_samples=10000, random_state=42):
    np.random.seed(random_state)
    
    n_male = n_samples // 2
    n_female = n_samples - n_male
    
    male_height = np.random.normal(loc=175.3, scale=7.1, size=n_male)
    male_weight = np.random.normal(loc=80.2, scale=12.4, size=n_male)
    
    female_height = np.random.normal(loc=162.1, scale=6.5, size=n_female)
    female_weight = np.random.normal(loc=65.3, scale=10.8, size=n_female)
    
    heights = np.concatenate([male_height, female_height])
    weights = np.concatenate([male_weight, female_weight])
    genders = np.array(['Male'] * n_male + ['Female'] * n_female)
    
    mask = (heights > 140) & (heights < 210) & (weights > 35) & (weights < 160)
    heights = heights[mask]
    weights = weights[mask]
    genders = genders[mask]
    
    height_bmi = heights / 100
    bmi = weights / (height_bmi ** 2)
    
    df = pd.DataFrame({
        'height_cm': np.round(heights, 1),
        'weight_kg': np.round(weights, 1),
        'bmi': np.round(bmi, 2),
        'gender': genders
    })
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

def save_datasets(df, train_size=0.8, random_state=42):
    train_df, test_df = train_test_split(df, test_size=1-train_size, 
                                         stratify=df['gender'], random_state=random_state)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print(f"Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")
    print(f"Train gender distribution:\n{train_df['gender'].value_counts()}")
    print(f"Test gender distribution:\n{test_df['gender'].value_counts()}")
    return train_df, test_df

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(n_samples=10000)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nOverall gender distribution:\n{df['gender'].value_counts()}")
    print(f"\nStatistics:\n{df.groupby('gender').describe()}")
    
    save_datasets(df)
