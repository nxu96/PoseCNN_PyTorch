import torch
import HoughVoting
print(torch.__version__)
a = torch.Tensor([[1,2],[3,4]])
print (a)
b = a.flatten()
print(b)

print(HoughVoting.forward)
# help(HoughVoting.forward)
