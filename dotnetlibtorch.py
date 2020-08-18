import torch

model = torch.nn.Linear(3, 6)

traced_model = torch.jit.trace_module(model, example_inputs = torch.rand(16, model.weight.shape[1]))
torch.jit.save(traced_model, 'jit_traced_model.pt')

scripted_model = torch.jit.script(model)
torch.jit.save(traced_model, 'jit_scripted_model.pt')
