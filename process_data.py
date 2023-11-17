import pandas as pd
import numpy as np

Data = pd.DataFrame()
for i in range(2016, 2021):
    DataURL1 = f'Data/players_raw_{i}.csv'
    Data1 = pd.read_csv(DataURL1)
    DataURL2 = f'Data/players_raw_{i+1}.csv'
    Data2 = pd.read_csv(DataURL2)
    # Remove players with injuries who missed portions of the season due to it, or loans
    Data1 = Data1[Data1['news'].isnull()]
    # Remove Columns of DataFrame we don't plan on using
    Data1 = Data1[['first_name', 'second_name',
                   'assists', 'element_type', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 'minutes', 'influence', 'points_per_game', 'saves', 'bonus', 'yellow_cards', 'total_points']]
    Data2 = Data2[Data2['news'].isnull()]
    # Merge the total points of the latter year to that of the former year
    merged_df = Data1.merge(Data2[['first_name', 'second_name', 'total_points']],
                            on=['first_name', 'second_name'],
                            suffixes=('', '_next'),
                            how='left')
    # Remove players who stopped playing after year 1
    merged_df = merged_df[merged_df['total_points_next'].isnull() == False]

    Data = Data._append(merged_df)
# Save the final DataFrame to a CSV file
player_names = Data[['first_name', 'second_name']].copy()
Data.drop(columns=['first_name', 'second_name'], inplace=True)
Data = Data.astype(int)
print(Data.dtypes)
player_names.to_csv('name_data.csv', index=False)
Data.to_csv('final_data.csv', index=False)
