import csv

class BBCDatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = []
        self.label=[]
    
    def load_dataset(self):
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            for row in reader:
                self.dataset.append(row[0])
                self.label.append(row[1])
    
        return self.dataset,self.label

# # Example usage
# dataset_path = 'path/to/bbc_dataset.csv'
# loader = BBCDatasetLoader(dataset_path)
# loader.load_dataset()
# bbc_dataset = loader.get_dataset()
