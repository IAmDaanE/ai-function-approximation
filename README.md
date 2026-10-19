# AI Function Approximation

Train an AI model to take the shape of any mathematical function.

---

<img width="1464" height="910" alt="Schermafbeelding 2026-10-18 171348" src="https://github.com/user-attachments/assets/eeef5c04-93dd-493c-8bb0-7b5ef6c3cda3" />

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
