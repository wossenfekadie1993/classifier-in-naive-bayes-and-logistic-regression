import random

class WeatherDatasetGenerator:
    def __init__(self, num_entries):
        self.num_entries = num_entries
        self.dataset = []
        self.label=[]
        
    def generate_weather_data(self):
        for _ in range(self.num_entries):
            date = self.generate_random_date()
            temperature = self.generate_random_temperature()
            humidity = self.generate_random_humidity()
            wind_speed = self.generate_random_wind_speed()
            
            weather_entry = {
                'date': random.randint(1, 31),
                'temperature': random.uniform(-20.0, 40.0),
                'humidity': random.uniform(0.0, 100.0),
                'wind_speed': random.uniform(0.0, 30.0)
            }
            if weather_entry['temperature']>30:
                self.label.append('Hot')
            elif weather_entry['temperature']<15:
                self.label.append('cold')
            else:
                self.label.append('normal')
            self.dataset.append(weather_entry)
        return self.dataset

# # Example usage
# num_entries = 10
# generator = WeatherDatasetGenerator(num_entries)
# generator.generate_weather_data()
# dataset = generator.get_dataset()

# # Accessing the weather data
# for entry in dataset:
#     print(entry)
