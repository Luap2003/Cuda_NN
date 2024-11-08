import urllib.request
import gzip
import os

def download_and_extract_mnist(destination_path):
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte.gz': 't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte.gz': 't10k-labels-idx1-ubyte'
    }

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for gz_filename, uncompressed_filename in files.items():
        gz_filepath = os.path.join(destination_path, gz_filename)
        uncompressed_filepath = os.path.join(destination_path, uncompressed_filename)

        # Download the file if it doesn't exist
        if not os.path.exists(gz_filepath):
            print(f'Downloading {gz_filename}...')
            urllib.request.urlretrieve(base_url + gz_filename, gz_filepath)
        else:
            print(f'{gz_filename} already exists.')

        # Unzip the file if the uncompressed file doesn't exist
        if not os.path.exists(uncompressed_filepath):
            print(f'Extracting {gz_filename}...')
            with gzip.open(gz_filepath, 'rb') as f_in:
                with open(uncompressed_filepath, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            print(f'{uncompressed_filename} already exists.')

if __name__ == '__main__':
    download_and_extract_mnist('mnist_data')
