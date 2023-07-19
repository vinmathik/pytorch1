import torch
print("torch version : {}".format(torch.__version__))
# Create a Tensor with just ones in a column
a = torch.ones(5)
# Print the tensor we created
print(a)
 # Create a Tensor with just zeros in a column
b = torch.zeros(5)
print(b)
c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)
d = torch.zeros(3,2)
print(d)
e = torch.ones(3,2)
print(e)
f = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
print(f)
# 3D Tensor
g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(g)
print(f.shape)
print(e.shape)
print(g.shape)