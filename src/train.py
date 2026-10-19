import sys
import barebones_ml as bbml 
import numpy as np
import pgplot
import pygame
import argparse

from utils import generate_sampled_x_values, get_full_x_values, function_stringparser

pygame.init()
screen = pygame.display.set_mode((980, 580))
pygame.display.set_caption("Function Approximation Training")

hidden_size = 64

network = bbml.Network("mse")

network.add(bbml.Layer(1, hidden_size, "relu", "random_small"))
network.add(bbml.Layer(hidden_size, hidden_size,"relu", "random_small"))
network.add(bbml.Layer(hidden_size, hidden_size, "relu", "random_small"))
network.add(bbml.Layer(hidden_size, 1, "linear", "xavier"))

vis = bbml.Visualization()
plotter = pgplot.Plotter(["smoothline", "scatter"], custom_colours=True, plot_colours_dict={"background": "#000000", "axes": "#ffffff", 0: "#ff9933", 1: "#0086d4"}, display_axes=True, display_legend=False)

def run_training(epochs, start_lr, function_string, samples):
    current_lr = start_lr

    x_min = -1
    x_max = 1

    raw_full_x_values = get_full_x_values(x_min, x_max, 400)
    raw_sampled_x_values = generate_sampled_x_values(x_min, x_max, samples)
    raw_sampled_y_values = function_stringparser(function_string, raw_sampled_x_values)

    sampled_x_values = raw_sampled_x_values.reshape(-1, 1)
    sampled_y_values = raw_sampled_y_values.reshape(-1, 1)
    full_x_values = raw_full_x_values.reshape(-1, 1)

    for epoch in range(epochs):
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
        network.epoch = epoch
        network.current_lr = current_lr
        predicted_y_values = network.forward(sampled_x_values)
        loss = network.loss_function(predicted_y_values, sampled_y_values)
        network.loss = loss
        network.backward(predicted_y_values, sampled_y_values)
        network.update(current_lr)
        current_lr = bbml.LRDecays.linear_decay(current_lr, 0.00001, 0.0005)
        if epoch % 10 == 0:
            full_y_prediction = network.forward(full_x_values)
            vis.full_draw_no_graph(network, screen)
            graph_surface = pygame.Surface((420, 420))
            plotter.plot(graph_surface, [[raw_full_x_values, full_y_prediction],[raw_sampled_x_values, sampled_y_values]])
            screen.blit(graph_surface, (540, 20))
            pygame.display.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a function approximation model."
    )

    parser.add_index = parser.add_argument(
            "function_string", 
            type=str, 
            help="The name of the function you want the model to approximate."
        )

    parser.add_index = parser.add_argument(
            "samples", 
            type=int, 
            help="The amount of sample points on the function the network receives as input."
        )

    args = parser.parse_args()
    run_training(50000, 0.5, args.function_string, args.samples)