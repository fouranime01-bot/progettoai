def test_imports():
    from myproject import model, training, predict, eda
    assert True


def test_model_forward():
    import torch
    from myproject.model import SimpleNet

    model = SimpleNet()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)

    assert out.shape == (1, 10)

