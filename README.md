# How to run the code 
`python main.py --bs=8 --steps=300 --eot=10 --num_test=100`

This code will:

1. Train a model using multi-resolution and cross-max self-ensemble on the CIFAR-100 dataset.
2. Perform an adaptive attack on the trained model by:
    - Running a PGD attack with the same model (without cross-max) and transferring the attack to the target model.
    - Selecting unsuccessful samples and running a PGD attack on the target model again with multiple restarts
3. Report the attack success rate compared to the auto-attack.

# A quick start

We provide a simple example in the `quick_start.ipynb` file. This example demonstrates how to load the trained model and test the attack success rate using our generated adversarial examples.

