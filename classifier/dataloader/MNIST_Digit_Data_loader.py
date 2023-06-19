from mnist import MNIST
class MNISTDigitDataLoader:
    def __init__(self,minist_data_path):
        self.minist_data_path=minist_data_path
        
    def Data_loader(self):
        mndata = MNIST(self.minist_data_path)
        train_images, train_labels = mndata.load_training()
        test_images, test_labels = mndata.load_testing()
        return train_images, train_labels, test_images, test_labels