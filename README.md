# AI Function Approximation

Train an AI to take the shape of any mathmematical function.

---

<img width="958" height="630" alt="image" src="https://github.com/user-attachments/assets/d580e826-7ec1-4a9a-afb8-221cf42fee04" />

## About the Project

This was written in python using my very own [barebones_ml](https://github.com/IAmDaanE/bare-bones-ml) machine learning library, this gives a nice UI overview of what is happening during training.  The function's are visualized using [pgplot](https://github.com/IAmDaanE/pygame-plot), a pygame graphing library I also wrote myself. The target function is sampled and these sample points are given as input to the network. It should then learn the function and give a y value for every x.

## Starting Visual Training

**Requires:** Python 3.10 - 3.14
1. Install the needed libraries, preferably in a venv.

    ```
    pip install -r requirements.txt
    ```

2. Start and watch the training

    ```
    python src/train.py "cos" 45
    ```

    There are two cli arguments: the name of the mathmetical function and the amount of data samples.

## License

This project is open-source and available under the MIT License.