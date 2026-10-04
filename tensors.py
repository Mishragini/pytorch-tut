import torch
import numpy as np

# directly
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# from np array
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)
print(x_np)

# from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3)
print(torch.rand(shape))
print(torch.ones(shape))
print(torch.zeros(shape))

print(x_rand.shape)
print(x_rand.dtype)
print(x_rand.device)

if torch.accelerator.is_available():
    tensor = x_rand.to(torch.accelerator.current_accelerator())
    print(tensor.device)
    
tensor = torch.rand(4,4)  
print(tensor)
print(tensor[0]) 
print(tensor[:,0])
print(tensor[:,-1])

t1 = torch.cat([tensor,tensor],dim=1)
print(t1)
agg = tensor.sum()
print(agg)
agg_item = agg.item()
print(agg_item,type(agg_item))