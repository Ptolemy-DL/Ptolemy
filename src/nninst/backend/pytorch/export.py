import torch.jit
from lenet import LeNet
from torch.autograd import Variable

from nninst.utils.fs import abspath

if __name__ == "__main__":
    dummy_input = Variable(torch.randn(1, 1, 28, 28))
    # trace, torch_out = torch.jit.trace(LeNet(), dummy_input)
    model = LeNet()
    # path = abspath("lenet_model.pth")
    # model.load_state_dict(torch.load(path))
    torch.onnx.export(model, dummy_input, abspath("lenet.onnx"))
