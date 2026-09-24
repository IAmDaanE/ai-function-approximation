# ML Function Approximation

Train an AI to take the shape of any mathmematical function.

---

<img width="958" height="630" alt="image" src="https://github.com/user-attachments/assets/d580e826-7ec1-4a9a-afb8-221cf42fee04" />

## About the Project

This was written in python using my very own [barebones_ml](https://github.com/IAmDaanE/bare-bones-ml) machine learning library. The input into the neural network is just the normalized x value and the network should spit out the y value for that x value. The function's are visualized using MatPlotLib and the neural network is visualized by barebones_ml using pygame.

## Starting Visual Training

**Requires:** Python 3.10 - 3.14
1. Install the needed libraries, preferably in a venv.

    ```
    pip install -r requirements.txt
    ```
2. Start and watch the training

    ```
    python train.py
    ```
3. To change the function the AI should learn change this line in train.py:

    ```python
    correct_y = np.sin(x / 15)
    ```

## License

This project is open-source and available under the MIT License.
