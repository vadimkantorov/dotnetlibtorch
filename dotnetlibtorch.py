import torch

model = torch.nn.Linear(8, 4)

example_input = torch.rand(16, 8)

traced_model = torch.jit.trace_module(model, example_input)
torch.jit.save(traced_model, 'traced_model.pt')

scripted_model = torch.jit.script(model)
torch.jit.save(traced_model, 'scripted_model.pt')
