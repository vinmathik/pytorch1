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
# Get element at index 2
print(c[2])
     # All indices starting from 0
 # Get element at row 1, column 0
print(f[1,0])
# We can also use the following
print(f[1][0])
# Similarly for 3D Tensor
print(g[1,0,0])
print(g[1][0][0])
int_tensor = torch.tensor([[1,2,3],[4,5,6]])
print(int_tensor.dtype)
 # What if we changed any one element to floating point number?
int_tensor = torch.tensor([[1,2,3],[4.,5,6]])
print(int_tensor.dtype)
print(int_tensor)
 # This can be overridden as follows
int_tensor = torch.tensor([[1,2,3],[4.,5,6]], dtype=torch.int32)
print(int_tensor.dtype)
print(int_tensor)