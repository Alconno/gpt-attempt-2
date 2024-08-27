import torch

# Create a tensor and transpose it
x = torch.randn(2, 3)
y = x.transpose(0, 1)
print(y)

print(y.is_contiguous())  # False, because transpose doesn't result in a contiguous tensor

# Making it contiguous
y_contiguous = y.contiguous()
print(y_contiguous.is_contiguous())  # True

print(y)