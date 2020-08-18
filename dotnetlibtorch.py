import torch

model = torch.nn.Linear(3, 6)

example_input = torch.rand(16, model.weight.shape[1])

traced_model = torch.jit.trace_module(model, example_input)
torch.jit.save(traced_model, 'jit_traced_model.pt')

scripted_model = torch.jit.script(model)
torch.jit.save(traced_model, 'jit_scripted_model.pt')
