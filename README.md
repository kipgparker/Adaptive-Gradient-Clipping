# Adaptive-Gradient-Clipping

Needs more testing

[Yannic Kilchers Video](https://www.youtube.com/watch?v=rNkHjZtH0RQ&ab_channel=YannicKilcher)

## Usage

```python
from agc import AGC

optimizer.zero_grad()        
loss, output = model(data)
loss.backward()

AGC(model.parameters(), args.clip)

optimizer.step()
```

# Citations

```bibtex
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```
