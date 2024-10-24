
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