import pickle

pickle_file_path = 'seeker_qtable.pickle'

with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

if isinstance(data, dict):
    for key, value in data.items():
        print(f'{key}: {value}')
