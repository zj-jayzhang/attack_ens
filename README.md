# How to run the code 
`python main.py --bs=8 --steps=300 --eot=10 --num_test=100`

This code will:

1. Train a model using multi-resolution and cross-max self-ensemble on the CIFAR-100 dataset.
2. Perform an adaptive attack on the trained model by:
    - Running a PGD attack with the same model (without cross-max) and transferring the attack to the target model.
    - Selecting unsuccessful samples and running a PGD attack on the target model again.
3. Report the attack success rate compared to the auto-attack.

# A quick start

We provide a simple example in the `quick_start.ipynb` file. This example demonstrates how to load the trained model and test the attack success rate using our generated adversarial examples.

The result is as follows:

```
With 100 clean examples, test for 10 times, clean accuracy: 62.40%±2.87%

setting parameters for rand version
using rand version including apgd-ce, apgd-dlr.
initial accuracy: 54.00%
apgd-ce - 1/2 - 11 out of 32 successfully perturbed
apgd-ce - 2/2 - 14 out of 22 successfully perturbed
robust accuracy after APGD-CE: 29.00% (total time 1371.9 s)
apgd-dlr - 1/1 - 3 out of 29 successfully perturbed
robust accuracy after APGD-DLR: 26.00% (total time 2221.5 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 26.00%

After auto attack, with 100 clean examples,  test for 10 times, robust accuracy: 48.50%±2.97%
===================================
===================================
After our attack, with 100 clean examples,  test for 10 times, robust accuracy: 9.70%±1.85%
Time cost:  37.4475524743398 mins
```