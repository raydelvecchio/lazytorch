import pickle
from nn import DenseLayer, FunctionApproximatorNN, MSE_Loss  # must import all of these so we can load the pickle!
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from lazytorch import Tensor
import matplotlib.pyplot as plt
import imageio
import glob

def generate_loss_graph(fann: FunctionApproximatorNN):
    plt.figure(figsize=(10, 5))
    plt.plot(fann.training_losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    os.makedirs('graphs', exist_ok=True)
    plt.savefig('graphs/training_loss.png')

def generate_comparison_graph(fann: FunctionApproximatorNN):
    x = [i * (20 / 999) - 10 for i in range(1000)]
    y_true = [i ** 2 for i in x]
    y_learned = [fann.forward(Tensor([[i]])).data[0][0].value for i in x]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true, label='True Function (x^2)', color='blue')
    plt.plot(x, y_learned, label='Learned Function', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of True Function and Learned Function')
    plt.legend()
    plt.grid(True)
    os.makedirs('graphs', exist_ok=True)
    plt.savefig('graphs/comparison_graph.png')

def generate_all_comparison_graphs():
    x = [i * (20 / 499) - 10 for i in range(500)]
    y_true = [i ** 2 for i in x]
    
    checkpoint_files = sorted(glob.glob('checkpoints/*.pkl'), key=lambda x: int(os.path.basename(x).split('_')[1]))
    image_files = []

    for checkpoint_file in checkpoint_files:
        with open(checkpoint_file, 'rb') as f:
            fann = pickle.load(f)

        y_learned = [fann.forward(Tensor([[i]])).data[0][0].value for i in x]

        plt.figure(figsize=(10, 5))
        plt.plot(x, y_true, color='blue')
        plt.plot(x, y_learned, color='red', linestyle='--')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Learned vs True - {checkpoint_file}')
        plt.grid(True)
        os.makedirs('graphs', exist_ok=True)
        graph_path = f'graphs/comparison_graph_{os.path.basename(checkpoint_file)}.png'
        plt.savefig(graph_path)
        image_files.append(graph_path)
        plt.close()

    images = [imageio.imread(image_file) for image_file in image_files]
    imageio.mimsave('graphs/comparison_graphs.gif', images, duration=2.25, loop=0)

if __name__ == "__main__":
    with open('checkpoints/epoch_20_checkpoint.pkl', 'rb') as f:
        fann = pickle.load(f)

    generate_loss_graph(fann)
    generate_all_comparison_graphs()
