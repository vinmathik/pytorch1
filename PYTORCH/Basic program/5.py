import torch
# Create tensor
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[-1,2,-3],[4,-5,6]])
 # Addition
print(tensor1+tensor2)
# We can also use
print(torch.add(tensor1,tensor2))
 # Subtraction
print(tensor1-tensor2)
# We can also use
print(torch.sub(tensor1,tensor2))
 # Multiplication
# Tensor with Scalar
print(tensor1 * 2)
 # Tensor with another tensor
# Elementwise Multiplication
print(tensor1 * tensor2)
 # Matrix multiplication
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
print(torch.mm(tensor1,tensor3))
 # Division
# Tensor with scalar
print(tensor1/2)
 # Tensor with another tensor
# Elementwise division
print(tensor1/tensor2)