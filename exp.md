
# 1. exp 1
> 32 clean examples, defense on
```markdown
With 32 clean examples, clean accuracy: 68.75%
setting parameters for rand version
using rand version including apgd-ce, apgd-dlr.
initial accuracy: 75.00%
apgd-ce - 1/1 - 10 out of 24 successfully perturbed
robust accuracy after APGD-CE: 43.75% (total time 600.0 s)
apgd-dlr - 1/1 - 4 out of 14 successfully perturbed
robust accuracy after APGD-DLR: 31.25% (total time 1097.7 s)
max Linf perturbation: 0.03137, nan in tensor: 0, max: 1.00000, min: 0.00000
robust accuracy: 31.25%
With 32 clean examples, robust accuracy: 46.88%
Time taken = 1100 seconds
```
# pgd20
> 200 and 1000 examples, defense on
```markdown
clean accuracy: 72.32%, robust accuracy: 61.61%
```

> 200 and 1000 examples, fixed randomness 
```markdown
clean accuracy: 60.27%, robust accuracy: 34.82%
clean accuracy: 56.93%, robust accuracy: 32.52%
```

> 200 and 1000 examples, ensemble off
```markdown
clean accuracy: 74.55%, robust accuracy: 38.84%
```


> 200 and 1000 examples, ensemble off + fixed randomness
```markdown
clean accuracy: 62.50%, robust accuracy: 4.46%
clean accuracy: 58.01%, robust accuracy: 2.83%
```

| seems interesting, 1000 examples, ensemble on + fixed randomness
> without cross-max
```markdown
clean accuracy: 58.50%, robust accuracy: 1.17%
```
> with cross-max
```markdown
clean accuracy: 56.93%, robust accuracy: 32.52%
```
=> cross-max is important!!!

| abalation study on cross-max, ensemble on + fixed randomness, 1000 examples, 

A:      `all_logits = all_logits - torch.max(all_logits, dim=2, keepdim=True).values`

B:
`all_logits = all_logits - torch.max(all_logits, dim=1, keepdim=True).values`

C: `logits = torch.topk(all_logits, 3, dim=1).values[:, 2]`
```
ABC:  clean accuracy: 56.93%, robust accuracy: 32.52%
C: clean accuracy: 55.18%, robust accuracy: 1.46
BC: clean accuracy: 0.78%, robust accuracy: 0.00%
AC: clean accuracy: 57.62%, robust accuracy: 8.69%

AB+avg: clean accuracy: 54.39%, robust accuracy: 4.30%
AB+cross-max: clean accuracy: 56.93%, robust accuracy: 32.52% 

```
# attack intermidiate layer

baseline: AB+cross-max: clean accuracy: 56.93%, robust accuracy: 32.52% 
> attack layer 20
```
clean accuracy: 56.93%, robust accuracy on source: 0.39%, robust accuracy on target: 21.68%
```

> attack layer 30
```
clean accuracy: 56.93%, robust accuracy on source: 0.39%, robust accuracy on target: 10.74%
```

> attack layer 40
```
clean accuracy: 56.93%, robust accuracy on source: 0.59%, robust accuracy on target: 5.66%
```

> attack layer 50
```
clean accuracy: 56.93%, robust accuracy on source: 1.17%, robust accuracy on target: 8.50%
```

> ens many layers
```
clean accuracy: 56.93%, robust accuracy on source: 1.17%, robust accuracy on target: 1.86%
```

> add randomness, ens many layers
baseline: clean accuracy: 66.31%, robust accuracy: 56.05%
```
clean accuracy: 67.19%, robust accuracy on source: 30.08%, robust accuracy on target: 30.08%
```
no, actually it's clean accuracy: 67.48%, robust accuracy on source: 1.27%, robust accuracy on target: 46.58%


# exp.2

we do class inheritance, now we have a class `TargetModel` and `SourceModel`, the first one turns on all defenses mechanisms, we use the second one to attack the model.

baseline: clean accuracy: 66.31%, robust accuracy: 56.05%

1. Setting: 1000 examples, fix_seed=True, turn off cross-max, ensemble on
clean accuracy: 67.48%, robust accuracy on source: 15.43%, robust accuracy on target: 49.51%

2. Setting: 1000 examples, fix_seed=False, turn on cross-max, ensemble on 
clean accuracy: 67.19%, robust accuracy on source: 34.28%, robust accuracy on target: 35.35%
clean accuracy: 67.19%, robust accuracy on source: 34.18%, robust accuracy on target: 35.35%
clean accuracy: 67.19%, robust accuracy on source: 34.18%, robust accuracy on target: 35.35%
==> do not fix seed, it is better for transfer attack???
btw, if I turn on cross-max for source model:
clean accuracy: 67.19%, robust accuracy on source: 54.88%, robust accuracy on target: 58.40%
==> which means averaging logits is strong enough to break this defense mechanism.


3. if I do 2, and ensemble it twice,
clean accuracy: 67.19%, robust accuracy on source: 41.99%, robust accuracy on target: 41.02%

4. what if I only use the 20-layer or 30-layer model?
20: clean accuracy: 67.19%, robust accuracy on source: 21.48%, robust accuracy on target: 52.93%
30: clean accuracy: 67.19%, robust accuracy on source: 29.30%, robust accuracy on target: 47.85%
50: clean accuracy: 67.19%, robust accuracy on source: 39.26%, robust accuracy on target: 50.29%
==> it's better to use ensemble model

one more baseline: clean accuracy: 67.68%, robust accuracy on source: 34.77%, robust accuracy on target: 36.91

> based on this baseline, we try eot:
1. 2 aug: clean accuracy: 68.65%, robust accuracy on source: 43.16%, robust accuracy on target: 45.80%
2. 2 raw: clean accuracy: 68.65%, robust accuracy on source: 30.96%, robust accuracy on target: 31.84%
3. 4 raw: clean accuracy: 69.00%, robust accuracy on source: 21.70%, robust accuracy on target: 32.30%
4. 6 raw: clean accuracy: 67.50%, robust accuracy on source: 20.90%, robust accuracy on target: 29.30%
5: 15 raw: clean accuracy: 69.70%, robust accuracy on source: 13.80%, robust accuracy on target: 31.00%


> pgd 40 seems not working well?
1 aug, clean accuracy: 69.10%, robust accuracy on source: 20.00%, robust accuracy on target: 37.70%

| + early stopping not working 
#steps:40, #eot: 2 | Nat Err: 317.0 | Rob_Source Err: 603.0 | Rob_Target Err: 333.0 | Total: 1000 
:
clean accuracy: 68.30%, robust accuracy on source: 39.70%, robust accuracy on target: 66.7

> redo it, 6 raw, pgd20, no early stopping


# exp.3
## test if cross-max is important
| a: for target model, fix seed, transfer attack
1. for source model, do not fix seed: 

    clean accuracy: 68.65%, robust accuracy on source: 35.35%, robust accuracy on target: 36.72%

2. for target model, fix seed, for source model, fix seed:

    clean accuracy: 68.65%, robust accuracy on source: 15.92%, robust accuracy on target: 16.41%

=> for transfer attack, cross-max still works a bit.


| a: for target model, fix seed, direct attack
clean accuracy: 68.65%, robust accuracy: 41.70%

=> seems the ensemble model works well for direct attack??

double check: what if not fix seed for target model?

| clean accuracy: 68.55%, robust accuracy: 57.91%
=> this seems right


## double check the best result we got
| 1 aug, pgd20, bs=32
clean accuracy: 68.65%, robust accuracy on source: 36.13%, robust accuracy on target: 37.70%

| 1 aug, pgd20, bs=48
clean accuracy: 68.55%, robust accuracy on source: 36.21%, robust accuracy on target: 37.90%

| 1 aug, pgd20, bs=1

#steps:20, #eot: 1 | Nat Err: 218.0 | Rob_Source Err: 519.0 | Rob_Target Err: 312.0 | Total: 692
=> it's weird, bs=1 seems not working well. why??

model.eval() ==> 

#steps:20, #eot: 1 | Nat Err: 33.0 | Rob_Source Err: 83.0 | Rob_Target Err: 46.0 | Total: 124
=> doesn't work well, seems the model is already in eval mode.

| 1 aug, pgd20, bs=2

=> doesn't work well

| 1 aug, pgd40, bs=48
clean accuracy: 68.25%, robust accuracy on source: 25.60%, robust accuracy on target: 25.99%

| 1 aug, pgd60, bs=48
clean accuracy: 68.85%, robust accuracy on source: 22.42%, robust accuracy on target: 23.21%

| 1 aug, pgd60, bs=16
clean accuracy: 70.14%, robust accuracy on source: 21.83%, robust accuracy on target: 24.01%

| 1 aug, pgd100, 
clean accuracy: 69.35%, robust accuracy on source: 20.24%, robust accuracy on target: 21.33%