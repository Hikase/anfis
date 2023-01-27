import torch

from anfis import create_anfis
from anfis.membership_functions import BellUniformlyBuilder

if __name__ == '__main__':
    data = torch.tensor([[5.1000, 3.5000, 1.4000],
                         [4.9000, 3.0000, 1.4000],
                         [4.7000, 3.2000, 1.3000],
                         [4.6000, 3.1000, 1.5000]])
    model = create_anfis(out_features=3,
                         min_values=torch.min(data.t(), dim=1).values.tolist(),
                         max_values=torch.max(data.t(), dim=1).values.tolist(),
                         membership_function_builder=BellUniformlyBuilder(),
                         n_classes=3)
    print(model(data))
