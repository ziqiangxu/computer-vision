from typing import List

import matplotlib.pyplot as plt
import numpy as np

plt.ion()


def show_loss(train_losses: List, valid_losses: List, save_name: str):
    plt.clf()
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.draw()
    plt.legend()
    if save_name:
        plt.savefig(save_name)
    plt.pause(0.0001)
