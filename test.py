import torch


for i in range(5):
    model = torch.load(f'stoch_depth_task_{i}.pth')
    print(model)
    print('\n\n\n\n')
    # exit()
