import argparse
import h5py
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, default='output.tflite')

args = parser.parse_args()


f = h5py.File(args.input, 'r+')
data_p = f.attrs['training_config']
data_p = data_p.decode().replace("learning_rate", "lr").encode()
f.attrs['training_config'] = data_p
f.close()

if __name__ == '__main__':
    tflite = tf.lite.TFLiteConverter.from_keras_model_file(args.input).convert()
    open(args.output, "wb").write(tflite)
